import torch
import argparse
import os
import logging
from tensorboardX import SummaryWriter
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from importlib import import_module
import PIL

parser = argparse.ArgumentParser(description='train scene clasification with places365')
parser.add_argument('--crop_size',default=224,type=int,help='crop input size')
parser.add_argument('--input_size', default=224, type=int, help='Resize input size')
parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7,8', type=str, help='identify gpus')
parser.add_argument('--workers', '-j', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of epochs to run')
parser.add_argument('--batch_size', '-b', default=128, type=int,
                    help='mini batch')
parser.add_argument('--learning_rate', '-lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--logger', default='training.log', type=str,
                    help='log file')
parser.add_argument('--log_freq', '-l', default=100, type=int,
                    help='print log')
parser.add_argument('--train_list', default='', type=str,
                    help='path to training list')
parser.add_argument('--val_list', default='', type=str,
                    help='path to validation list')
parser.add_argument('--train_rootdir',default='',type=str,help='the rootdir of train dataset')
parser.add_argument('--val_rootdir',default='',type=str,help='the rootdir of validation dataset')
parser.add_argument('--save_path', default='', type=str,
                    help='path to save checkpoint and log')
parser.add_argument('--pretrained_model', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--num_classes',type=int,help='num classes')
parser.add_argument('--network',help='the network of the model')
parser.add_argument('--lr_decay_step',type=int,default=10,help='reduce the learning rate every step')
parser.add_argument('--min_lr',type=float,default=1e-4,help='the min learning rate')
parser.add_argument('--is_use_memcache',default='',help='wether to use the memort cache')
parser.add_argument('--is_Scheduler_dropblock',help='wether to use Scheduler dropblock')
parser.add_argument('--is_cutout',help='wether to use the cutout')

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.queue = np.zeros(shape=[100,])
        self.index=0
        self.queue_avg=0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.queue[self.index%100]=self.val
        self.index+=1
        self.queue_avg=np.mean(self.queue)

def str2bool(v):
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_checkpoint(model):
    start_epoch=0
    best_loss=1e6
    if args.pretrained_model:
        if not os.path.isfile(args.pretrained_model):
            logger.info("=> no checkpoint found at '{}'".format(args.pretrained_model))
            return
        logger.info('=> load checkpoint {}'.format(args.pretrained_model))
        checkpoint=torch.load(args.pretrained_model)
        if 'epoch' in checkpoint.keys():
            start_epoch=checkpoint['epoch']
        if 'loss' in checkpoint.keys():
            best_loss=checkpoint['loss']
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        if 'network' in checkpoint.keys():
            model.load_state_dict(checkpoint['network'])
        logger.info('=> loaded checkpoint {} (epoch {}) (best_loss {})'.format(args.pretrained_model,start_epoch,best_loss))
    return start_epoch,best_loss

def check_parameters(model):
    #check the parameters
    for param in model.parameters():
        print(param.shape,param.requires_grad)

def save_checkpoint(model,epoch,loss,best_loss):

    model_path=os.path.join(args.save_path,'seresnet50_ckpt_epoch%d.pth'%epoch)
    torch.save({
        'epoch':epoch+1,
        'loss':loss,
        'state_dict':model.state_dict(),},model_path)

    model_path=os.path.join(args.save_path,'seresnet50_best_ckpt.pth')
    if loss<best_loss or not os.path.exists(model_path):
        best_loss=loss
        torch.save({
            'epoch': epoch + 1,
            'loss': best_loss,
            'state_dict': model.state_dict()},model_path)
    return best_loss

def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def logging_system():
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s')

    sysh = logging.StreamHandler()
    sysh.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(args.save_path, args.logger), 'w')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sysh)
    return logger

def train(train_loader, model, criterion, optimizer, epoch, log_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1= AverageMeter()
    top5= AverageMeter()

    logger.info("Training....")
    model.train()
    end = time.time()
    for iter_batch, sample_batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        img_batch = sample_batch[0].cuda()
        label_batch= sample_batch[1].cuda()
        # print(img_batch.shape,label_batch.shape)
        output=model(img_batch)
        loss = criterion(output, label_batch)
        prec = accuracy(output.data, label_batch.data, topk=(1,5))
        log_writer.add_scalar('Train_loss', loss, epoch*len(train_loader)+iter_batch)
        log_writer.add_scalar('train_top1',prec[0],epoch*len(train_loader)+iter_batch)
        log_writer.add_scalar('train_top5',prec[1],epoch*len(train_loader)+iter_batch)
        losses.update(loss.item(), img_batch.size(0))
        top1.update(prec[0], output.size(0))
        top5.update(prec[1], output.size(0))

        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_batch % args.log_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} [{loss.queue_avg:.4f},{loss.avg:.4f}]\t'
                        'Top1 {top1.val:.4f} ({top1.avg:.4f})\t'
                        'Top5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, iter_batch, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,top1=top1,top5=top5))

def val(val_loader,model, criterion,log_writer,epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    logger.info("valing....")
    model.eval()

    for iter_batch, sample_batch in enumerate(val_loader):

        img_batch = sample_batch[0].cuda()
        label_batch = sample_batch[1].cuda()
        # print(img_batch.shape, label_batch.shape)
        output = model(img_batch)
        loss = criterion(output, label_batch)
        losses.update(loss.item(), img_batch.size(0))
        prec = accuracy(output.data, label_batch.data, topk=(1,5))
        top1.update(prec[0], output.size(0))
        top5.update(prec[1], output.size(0))

    logger.info('Loss {loss.avg:.8f}\t'
                'Top1 {top1.avg:.4f}\t'
                'Top5 {top5.avg:.4f}\t'.format(loss=losses, top1=top1, top5=top5))

    if log_writer != None:
        log_writer.add_scalar('val_loss', losses.avg, epoch)
        log_writer.add_scalar('val_top1', top1.avg, epoch)
        log_writer.add_scalar('val_top5', top5.avg, epoch)

    return losses.avg

def adjust_learning_rate(optimizer, epoch,log_writer):
    lr = args.learning_rate * (0.1**(int(epoch/args.lr_decay_step)))
    # lr = args.learning_rate * (0.1**(epoch/10))
    lr = args.min_lr if lr <= args.min_lr else lr
    if epoch % (args.lr_decay_step) ==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    logger.info('learning rate: %f'%lr)
    log_writer.add_scalar('learning_rate',lr,epoch)

def Scheduler_dropblock(model, epoch, logger,log_writer):
    model.module.Linear_dropblock.step()
    keep_prob = model.module.Linear_dropblock.dropblock.keep_prob
    logger.info('keep_prob of dropblock: %s'%keep_prob)
    log_writer.add_scalar('keep_prob',keep_prob,epoch)

def main():

    from data.places365 import dataset_places365 as dataset
    from data.augmentation import Cutout,GaussianBlur

    augmentation_list = [transforms.Resize((256,256),interpolation=2),
                            GaussianBlur(p=0.3,radius=3),
                            transforms.RandomCrop((args.crop_size,args.crop_size)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                            transforms.RandomGrayscale(p=0.1)]

    transform_list = [transforms.Resize((args.input_size,args.input_size),interpolation=2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    Additional_list=[]
    if str2bool(args.is_cutout):
        Additional_list.append(Cutout(n_holes=1, length=32))

    train_transform=transforms.Compose(augmentation_list+transform_list+Additional_list)

    train_dst = dataset(args.train_list,args.train_rootdir,train_transform,is_use_memcache=str2bool(args.is_use_memcache))
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_transform=transforms.Compose(transform_list)
    val_dst = dataset(args.val_list, args.val_rootdir , val_transform, is_use_memcache=str2bool(args.is_use_memcache))
    val_loader =DataLoader(val_dst,batch_size=128,shuffle=False,num_workers=args.workers,pin_memory=False)
    
    start_epoch=0
    best_loss = 1e6

    module_name,method_name=args.network.split('.')
    module = import_module('network.%s'%module_name)
    method = getattr(module,method_name)
    model = method(num_classes=args.num_classes)
    model = nn.DataParallel(model).cuda()
    start_epoch,best_loss= load_checkpoint(model)
    # check_parameters(model)

    criterion = nn.CrossEntropyLoss()
    # criterion = ArcFace(margin=0.5,scale=64)
    criterion = criterion.cuda()
    optimizer = optim.SGD(filter(lambda p:p.requires_grad,model.parameters()),lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    log_writer = SummaryWriter(os.path.join(args.save_path, 'Training_log'))

    loss=val(val_loader,model,criterion,None,None)
    for epoch in range(start_epoch, start_epoch+args.epochs):

        if str2bool(args.is_Scheduler_dropblock):
            Scheduler_dropblock(model,epoch,logger,log_writer)

        adjust_learning_rate(optimizer,epoch,log_writer)
        train(train_loader, model, criterion, optimizer, epoch, log_writer)
        loss=val(val_loader,model,criterion,log_writer,epoch)
        best_loss=save_checkpoint(model,epoch,loss,best_loss)

if __name__ == '__main__':
    global args,logger,DCFNet
    
    args = parser.parse_args()
    logger = logging_system()
    logger.info(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.device_count()>1:
        logger.info('%d GPU found'%torch.cuda.device_count())
    main()
