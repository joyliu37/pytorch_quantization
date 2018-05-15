'''
Modified from https://github.com/pytorch/vision.git
'''
import math
from Model.quantization import *
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
__all__ = [
    'FixedVGG', 'fixed_vgg11', 'fixed_vgg11_bn', 'fixed_vgg13', 'fixed_vgg13_bn', 'fixed_vgg16', 'fixed_vgg16_bn',
    'fixed_vgg19_bn', 'fixed_vgg19',
]


class FixedVGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(FixedVGG, self).__init__()
        self.features = features
        idx = 0
        self.classifier = nn.Sequential(OrderedDict([
            (str(idx),nn.Dropout()),
            ('Q'+str(idx+1),activation_quantization()),
            (str(idx+1),nn.Linear(512, 512)),
            (str(idx+2),nn.ReLU(True)),
            (str(idx+3),nn.Dropout()),
            ('Q'+str(idx+2),activation_quantization() ),
            (str(idx+4),nn.Linear(512, 512)),
            (str(idx+5),nn.ReLU(True)),
            ('Q'+str(idx+3),activation_quantization() ),
            (str(idx+6),nn.Linear(512, 10))
            ])
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):

    layers = []
    idx = 0
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [(str(idx),nn.MaxPool2d(kernel_size=2, stride=2))]
            idx += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                #layers += [('Q'+str(idx),activation_quantization())]
                layers += [(str(idx),conv2d), (str(idx+1),nn.BatchNorm2d(v)), (str(idx+2),nn.ReLU(inplace=True))]
                idx += 3
            else:
                #if(idx < 17):
                layers += [('Q'+str(idx),activation_quantization())]
                layers += [(str(idx),conv2d), (str(idx+1),nn.ReLU(inplace=True))]
                idx += 2
            in_channels = v
    return nn.Sequential(OrderedDict(layers))


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def fixed_vgg11():
    """VGG 11-layer model (configuration "A")"""
    return FixedVGG(make_layers(cfg['A']))


def fixed_vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return FixedVGG(make_layers(cfg['A'], batch_norm=True))


def fixed_vgg13():
    """VGG 13-layer model (configuration "B")"""
    return FixedVGG(make_layers(cfg['B']))


def fixed_vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return FixedVGG(make_layers(cfg['B'], batch_norm=True))


def fixed_vgg16():
    """VGG 16-layer model (configuration "D")"""
    return FixedVGG(make_layers(cfg['D']))


def fixed_vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return FixedVGG(make_layers(cfg['D'], batch_norm=True))


def fixed_vgg19():
    """VGG 19-layer model (configuration "E")"""
    return FixedVGG(make_layers(cfg['E']))


def fixed_vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return FixedVGG(make_layers(cfg['E'], batch_norm=True))
