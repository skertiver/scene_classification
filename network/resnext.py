import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, withfactor, stride=1, cardinality=32, base_width=4, se_block=True, se_reduction=16):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * base_width *withfactor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.se_block = se_block

        if self.se_block:
            self.se = SEModule(out_channels, se_reduction)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        if self.se_block:
            bottleneck= self.se(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)

class SE_resNext(nn.Module):

    def __init__(self,  block, layers, num_classes=1000,is_evaluate=False):
        super(SE_resNext, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, withfactor=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, withfactor=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, withfactor=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, withfactor=8)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.is_evaluate = is_evaluate
        if self.is_evaluate:
            self.fc = nn.Conv2d(512 * block.expansion, num_classes,kernel_size=1,stride=1,bias=True)
            self.gloabal_avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,withfactor=1):
        # in_channels, out_channels, stride, cardinality, base_width, se_block=True, se_reduction=16
        layers = []
        layers.append(block(self.inplanes, planes * block.expansion, withfactor=withfactor , stride=stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes * block.expansion, withfactor=withfactor))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        if self.is_evaluate:
            x = self.fc(x)
            x = self.gloabal_avgpool(x)
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def se_resnext50(pretrained=False, **kwargs):

    model = SE_resNext(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def export_jni():
    from collections import OrderedDict

    model = se_resnext50(num_classes=365,is_evaluate=True)
    checkpoint = torch.load('/world/jiacongliao/model/places365/se_resnext50/se_resnext50_aug/seresnet50_best_ckpt.pth')
    
    param_state_dict=checkpoint['state_dict']

    new_state_dict = OrderedDict()
    fc_weight=param_state_dict['module.fc.weight']
    param_state_dict['module.fc.weight']=fc_weight.view(fc_weight.shape[0],fc_weight.shape[1],1,1)
    for key in param_state_dict:
        new_key = key.replace('module.','')
        new_state_dict[new_key]=param_state_dict[key]

    model.load_state_dict(new_state_dict)
    # model = model.module
    model.eval()
    dummy_input = torch.rand(1, 3, 256, 256)

    traced_net = torch.jit.trace(model, dummy_input)
    traced_net.save('places365_ResNeXt.pt')

def eval():

    import torch
    import cv2
    import numpy as np
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    from collections import OrderedDict

    # model = se_resnext50(num_classes=365,is_evaluate=True)
    # model = nn.DataParallel(model).cuda(8)
    # checkpoint = torch.load('/world/jiacongliao/model/places365/se_resnext50/se_resnext50_aug/seresnet50_best_ckpt.pth')
    
    # param_state_dict=checkpoint['state_dict']
    # fc_weight=param_state_dict['module.fc.weight']
    # param_state_dict['module.fc.weight']=fc_weight.view(fc_weight.shape[0],fc_weight.shape[1],1,1)
    # model.load_state_dict(param_state_dict)
    # model.eval()


    model = se_resnext50(num_classes=365,is_evaluate=True)
    checkpoint = torch.load('/world/jiacongliao/model/places365/se_resnext50/se_resnext50_aug/seresnet50_best_ckpt.pth')
    
    param_state_dict=checkpoint['state_dict']
    new_state_dict = OrderedDict()
    fc_weight=param_state_dict['module.fc.weight']
    param_state_dict['module.fc.weight']=fc_weight.view(fc_weight.shape[0],fc_weight.shape[1],1,1)
    for key in param_state_dict:
        new_key = key.replace('module.','')
        new_state_dict[new_key]=param_state_dict[key]

    model.load_state_dict(new_state_dict)
    # model = model.module
    model.eval()


    # transform_list = [transforms.Resize((256,256),interpolation=2),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    # val_transform=transforms.Compose(transform_list)
    # img = Image.open('/world/jiacongliao/place365/scene_classification/data/test3.jpg')
    # img = val_transform(img).view(1,3,256,256)

    img = cv2.imread('/world/jiacongliao/place365/scene_classification/data/test3.jpg')
    img = cv2.resize(img,(256,256))
    img = img / 255.0
    img = np.transpose(img, [2, 0, 1])
    mean =  np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = (img- mean) / std
    img = np.reshape(img,(1,3,img.shape[1],img.shape[2]))
    output=model(torch.from_numpy(img).float())
    output = F.softmax(output)
    print(np.argsort(output.cpu().data.numpy()))
    score, pred = output.topk(5, 1, True, True)
    print(score,pred)
    # print (output.shape)


if __name__ == '__main__':
    # model =  se_resnext50(num_classes=365)
    # print(model)
    # out=model(torch.rand(1,3,224,224))
    # print(out.shape)

    # export_jni()
    eval()

