import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

class CANNet2s(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet2s, self).__init__()
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 1024, batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        self.relu = nn.ReLU()
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self,x_prev,x):
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)

        x_prev = self.context(x_prev)
        x = self.context(x)

        x = torch.cat((x_prev,x),1)

        x = self.backend(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x

    def predict(self, x, x_prev):

        with torch.no_grad():
            prev_flow = self(x_prev, x)
            prev_flow_inverse = self(x, x_prev)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry_cuda = mask_boundry.cuda()
        mask_boundry_var = Variable(mask_boundry_cuda)

        reconstruction_from_prev = F.pad(prev_flow[:, 0, 1:, 1:], (0, 1, 0, 1)) \
                                   + F.pad(prev_flow[:, 1, 1:, :], (0, 0, 0, 1))\
                                   + F.pad(prev_flow[:, 2, 1:, :-1], (1, 0, 0, 1)) \
                                   + F.pad(prev_flow[:, 3, :, 1:], (0, 1, 0, 0))\
                                   + prev_flow[:, 4, :, :]\
                                   + F.pad(prev_flow[:, 5, :, :-1], (1, 0, 0, 0))\
                                   + F.pad(prev_flow[:, 6, :-1, 1:], (0, 1, 1, 0))\
                                   + F.pad(prev_flow[:, 7, :-1, :], (0, 0, 1, 0))\
                                   + F.pad(prev_flow[:, 8, :-1, :-1], (1, 0, 1, 0))\
                                   + prev_flow[:, 9, :, :] * mask_boundry_var

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[:, :9, :, :], dim=1) \
                                         + prev_flow_inverse[:, 9, :, :] * mask_boundry_var

        overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0)

        return overall.sum(dim=1).sum(dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load(self, filepath, testing=True):
        if testing:
            self.eval()
        checkpoint = torch.load(filepath)

        self.load_state_dict(checkpoint['state_dict'])


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
