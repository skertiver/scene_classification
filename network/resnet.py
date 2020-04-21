import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_fcn(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_fcn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Conv2d(512 * block.expansion, num_classes,kernel_size=1,stride=1,bias=True)
        self.gloabal_avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = self.fc(x)
        x = self.gloabal_avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class ResNet_split(nn.Module):

    def __init__(self, block, layers, num_classes=1000,is_evaluate=False):
        self.inplanes = 64
        super(ResNet_split, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer_1(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

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

    def _make_layer_1(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(128, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(128, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = F.pad(x, (0L, 1L, 0L, 1L), value=float('-inf'))
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



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet50_fcn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fcn(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def places_cnn_365(**kwargs):
    model = ResNet_split(Bottleneck, [8, 12, 23, 3],**kwargs)

    return model


def paser_param():
    import numpy as np
    from collections import OrderedDict
    import torch
    import re
    from pytorch_resnet import KitModel

    weight_file = '/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/pytorch_resnet.npy'
    weights_dict = np.load(weight_file, encoding='bytes').item()
    # print(sorted(weights_dict.keys()))
    new_state_dict = OrderedDict()

    for key in sorted(weights_dict.keys()):
        layer_name = key
        # print('----->',layer_name)
        if layer_name == 'conv1_1_3x3_s2':
            new_state_dict['conv1.weight'] = torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['bn1.weight'] = torch.from_numpy(weights_dict['conv1_1_3x3_s2/bn']['scale'])
            new_state_dict['bn1.bias'] = torch.from_numpy(weights_dict['conv1_1_3x3_s2/bn']['bias'])
            new_state_dict['bn1.running_mean'] = torch.from_numpy(weights_dict['conv1_1_3x3_s2/bn']['mean'])
            new_state_dict['bn1.running_var'] = torch.from_numpy(weights_dict['conv1_1_3x3_s2/bn']['var'])
            # new_state_dict['bn1.num_batches_tracked'] = torch.tensor(88075)


        elif layer_name == 'conv1_2_3x3':
            new_state_dict['conv2.weight'] = torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['bn2.weight'] = torch.from_numpy(weights_dict['conv1_2_3x3/bn']['scale'])
            new_state_dict['bn2.bias'] = torch.from_numpy(weights_dict['conv1_1_3x3_s2/bn']['bias'])
            new_state_dict['bn2.running_mean'] = torch.from_numpy(weights_dict['conv1_2_3x3/bn']['mean'])
            new_state_dict['bn2.running_var'] = torch.from_numpy(weights_dict['conv1_2_3x3/bn']['var'])
            # new_state_dict['bn2.num_batches_tracked'] = torch.tensor(88075)


        elif layer_name == 'conv1_3_3x3':
            new_state_dict['conv3.weight'] = torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['bn3.weight'] = torch.from_numpy(weights_dict['conv1_3_3x3/bn']['scale'])
            new_state_dict['bn3.bias'] = torch.from_numpy(weights_dict['conv1_3_3x3/bn']['bias'])
            new_state_dict['bn3.running_mean'] = torch.from_numpy(weights_dict['conv1_3_3x3/bn']['mean'])
            new_state_dict['bn3.running_var'] = torch.from_numpy(weights_dict['conv1_3_3x3/bn']['var'])
            # new_state_dict['bn3.num_batches_tracked'] = torch.tensor(88075)


        elif re.match(r'^conv\d_\d+_1x1_reduce$', layer_name):
            layer_num = int(layer_name[4])-1
            try:
                block_num = int(layer_name[6:8])-1
            except ValueError:
                block_num = int(layer_name[6])-1

            new_state_dict['layer%d.%d.conv1.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['layer%d.%d.bn1.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['scale'])
            new_state_dict['layer%d.%d.bn1.bias'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['bias'])
            new_state_dict['layer%d.%d.bn1.running_mean'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['mean'])
            new_state_dict['layer%d.%d.bn1.running_var'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['var'])
            # new_state_dict['layer%d.%d.bn1.num_batches_tracked'%(layer_num,block_num)] = torch.tensor(88075)


        elif re.match(r'^conv\d_\d+_3x3$', layer_name):
            layer_num = int(layer_name[4])-1
            try:
                block_num = int(layer_name[6:8])-1
            except ValueError:
                block_num = int(layer_name[6])-1

            new_state_dict['layer%d.%d.conv2.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['layer%d.%d.bn2.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['scale'])
            new_state_dict['layer%d.%d.bn2.bias'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['bias'])
            new_state_dict['layer%d.%d.bn2.running_mean'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['mean'])
            new_state_dict['layer%d.%d.bn2.running_var'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['var'])
            # new_state_dict['layer%d.%d.bn2.num_batches_tracked'%(layer_num,block_num)] = torch.tensor(88075)

        elif re.match(r'^conv\d_\d+_1x1_increase$', layer_name):
            layer_num = int(layer_name[4])-1
            try:
                block_num = int(layer_name[6:8])-1
            except ValueError:
                block_num = int(layer_name[6])-1

            new_state_dict['layer%d.%d.conv3.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['layer%d.%d.bn3.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['scale'])
            new_state_dict['layer%d.%d.bn3.bias'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['bias'])
            new_state_dict['layer%d.%d.bn3.running_mean'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['mean'])
            new_state_dict['layer%d.%d.bn3.running_var'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['var'])
            # new_state_dict['layer%d.%d.bn3.num_batches_tracked'%(layer_num,block_num)] = torch.tensor(88075)

        elif re.match(r'^conv\d_\d+_1x1_proj$', layer_name):
            layer_num = int(layer_name[4])-1
            try:
                block_num = int(layer_name[6:8])-1
            except ValueError:
                block_num = int(layer_name[6])-1
            #layer4.0.downsample.1.bias
            new_state_dict['layer%d.%d.downsample.0.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['layer%d.%d.downsample.1.weight'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['scale'])
            new_state_dict['layer%d.%d.downsample.1.bias'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['bias'])
            new_state_dict['layer%d.%d.downsample.1.running_mean'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['mean'])
            new_state_dict['layer%d.%d.downsample.1.running_var'%(layer_num,block_num)]= torch.from_numpy(weights_dict[layer_name+'/bn']['var'])
            # new_state_dict['layer%d.%d.downsample.1.num_batches_tracked'%(layer_num,block_num)] = torch.tensor(88075)

        elif layer_name == 'classifier_1':
            new_state_dict['fc.weight'] = torch.from_numpy(weights_dict[layer_name]['weights'])
            new_state_dict['fc.bias'] = torch.from_numpy(weights_dict[layer_name]['bias'])

        elif not re.match(r'^conv.*/bn$', layer_name):
            print(layer_name)
            pass

    model1 = places_cnn_365(num_classes=365)
    model1.load_state_dict(new_state_dict)
    out1 = model1(0.5*torch.ones(1,3,224,224))
    # print(out1)

    model2 = KitModel('/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/pytorch_resnet.npy')
    out2 =model2(0.5*torch.ones(1,3,224,224))
    # print(out2)

    print(out1 - out2)

    # torch.save({
    #         'name': 'Places2-365-CNN',
    #         'state_dict': model.state_dict()},'/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/Places365_resnet.pth')

    # for key in model.state_dict():
    #     print(key)

def plot_network():
    import torch
    from tensorboardX import SummaryWriter
    dummy_input = torch.rand(1, 3, 224, 224)
    model = places_cnn_365(num_classes=365)
    torch.onnx.export(model, dummy_input, "ResNet.onnx", verbose=True)
    with SummaryWriter(comment='resnet') as w:
        w.add_graph(model, (dummy_input, ))

def convert2onnx():
    from pytorch_resnet import KitModel
    import torch
    import onnx

    model = KitModel('/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/pytorch_resnet.npy')
    dummy_input = torch.rand(1, 3, 256, 256)
    torch.onnx.export(model, dummy_input, "places365_ResNet.onnx", verbose=False)

    model = onnx.load("places365_ResNet.onnx")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)

def export2jit():
    from pytorch_resnet import KitModel
    import torch

    model = KitModel('/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/pytorch_resnet.npy')
    dummy_input = torch.rand(1, 3, 256, 256)
    model.eval()

    traced_net = torch.jit.trace(model, dummy_input)
    traced_net.save('places365_ResNet.pt')

def eval():
    from pytorch_resnet import KitModel
    import torch
    import cv2
    import numpy as np
    import torch.nn.functional as F

    model = KitModel('/world/jiacongliao/model/places365/public/Places2-CNNs/Places2-365-CNN/pytorch_resnet.npy').cuda(9)
    model.eval()
    img = cv2.imread('/world/jiacongliao/place365/scene_classification/data/test3.jpg')
    img = cv2.resize(img,(256,256),cv2.INTER_LINEAR)
    img = img - np.array([105.448, 113.768, 116.052])
    # img = img - np.reshape(np.array([105.448, 113.768, 116.052]),[3, 1, 1])
    img = np.transpose(img, [2, 0, 1])
    img = np.reshape(img,(1,3,img.shape[1],img.shape[2]))
    output=model(torch.from_numpy(img).float().cuda(9))
    output = F.softmax(output)
    print(np.argsort(output.cpu().data.numpy()))
    score, pred = output.topk(5, 1, True, True)
    print(score,pred)
    # print (output.shape)


if __name__ == '__main__':
    # convert2onnx()
    eval()
    # export2jit()
