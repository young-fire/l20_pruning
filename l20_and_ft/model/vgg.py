
import torch.nn as nn
from collections import OrderedDict

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39,42]
class VGG(nn.Module):
    def __init__(self, layer_cfg, cfg=None, num_classes=10):
        super(VGG, self).__init__()

        if cfg is None:
            cfg = defaultcfg
        self.relucfg = relucfg

        self.layer_cfg = layer_cfg

        self.features = self._make_layers(cfg)
        
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(layer_cfg[-1], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))
        # self.a = 1
    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt=0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                x = self.layer_cfg[cnt]

                cnt+=1
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_16_bn(layer_cfg=None):
    if layer_cfg==None:
        layer_cfg =[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
 
    return VGG(layer_cfg=layer_cfg)

# vgg_16_bn = vgg_16_bn([0.45]*7+[0.78]*5)