import torch
import argparse
import os
import logging
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from importlib import import_module

parser = argparse.ArgumentParser(description='evaluate scene clasification with places365')
parser.add_argument('--input_size', default=224, type=int, help='crop input size')
parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7,8', type=str, help='identify gpus')
parser.add_argument('--workers', '-j', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--batch_size', '-b', default=128, type=int,
                    help='mini batch')
parser.add_argument('--logger', default='training.log', type=str,
                    help='log file')
parser.add_argument('--val_list', default='', type=str,
                    help='path to validation list')
parser.add_argument('--val_rootdir',default='',type=str,help='the rootdir of validation dataset')
parser.add_argument('--save_path', default='', type=str,
                    help='path to save checkpoint and log')
parser.add_argument('--pretrained_model', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--num_classes',type=int,help='num classes')
parser.add_argument('--network',help='the network of the model')
parser.add_argument('--convert2fcn',help='wether to convert model to fcn')

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
            param_state_dict=checkpoint['state_dict']
            if str2bool(args.convert2fcn):
                fc_weight=param_state_dict['module.fc.weight']
                param_state_dict['module.fc.weight']=fc_weight.view(fc_weight.shape[0],fc_weight.shape[1],1,1)
            model.load_state_dict(param_state_dict)
        if 'network' in checkpoint.keys():
            model.load_state_dict(checkpoint['network'])
        logger.info('=> loaded checkpoint {} (epoch {}) (best_loss {})'.format(args.pretrained_model,start_epoch,best_loss))
    return start_epoch,best_loss

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

def val(val_loader,model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    logger.info("valing....")
    model.eval()

    for iter_batch, sample_batch in enumerate(val_loader):

        img_batch = sample_batch[0].cuda()
        label_batch = sample_batch[1].cuda()
        output = model(img_batch)
        loss = criterion(output, label_batch)
        losses.update(loss.item(), img_batch.size(0))
        prec = accuracy(output.data, label_batch.data, topk=(1,5))
        top1.update(prec[0], output.size(0))
        top5.update(prec[1], output.size(0))

    logger.info('Loss {loss.avg:.8f}\t'
                'Top1 {top1.avg:.4f}\t'
                'Top5 {top5.avg:.4f}\t'.format(loss=losses, top1=top1, top5=top5))

    return losses.avg

def main():

    from data.places365 import dataset_places365 as dataset

    transform_list = [transforms.Resize((args.input_size,args.input_size),interpolation=2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]

    val_transform=transforms.Compose(transform_list)
    val_dst = dataset(args.val_list, args.val_rootdir , val_transform, is_use_memcache=False)
    val_loader =DataLoader(val_dst,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=False)

    module_name,method_name=args.network.split('.')
    module = import_module('network.%s'%module_name)
    method = getattr(module,method_name)
    model = method(num_classes=args.num_classes,is_evaluate=True)
    model = nn.DataParallel(model).cuda()
    start_epoch,_ = load_checkpoint(model)

    criterion = nn.CrossEntropyLoss()
    # criterion = ArcFace(margin=0.5,scale=64)
    criterion = criterion.cuda()

    val(val_loader,model,criterion)

if __name__ == '__main__':
    global args,logger,DCFNet
    
    args = parser.parse_args()
    logger = logging_system()
    logger.info(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.device_count()>1:
        logger.info('%d GPU found'%torch.cuda.device_count())
    main()