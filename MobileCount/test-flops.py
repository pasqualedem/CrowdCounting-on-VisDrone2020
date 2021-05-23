from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
import torchvision.models as tmodels
from config import cfg
from misc.utils import *
from models.ptflops import get_model_complexity_info

from torchvision.models import vgg16, vgg19, vgg11
"""
from torchvision.models import MCNN
from torchvision.models.VGG import VGG
from torchvision.models.VGG_decoder import VGG_decoder
from torchvision.models.gcy import ResNetLW
from torchvision.models.DeepLab_v3_2 import DeepLabv3
from torchvision.models.LWRF_MobileNetv2 import MobileNetLWRF
from torchvision.models.LWRF_ShuffleNetv2 import ShuffleNetLWRF
from torchvision.models.CSRNet import CSRNet
from torchvision.models.Res50 import Res50
from torchvision.models.MobileNetv2 import MobileNetV2
from torchvision.models.CMTL import CMTL
from torchvision.models.SANet import SANet
from torchvision.models.FPN import FPN
from torchvision.models.MobileNetv2_org_4 import MobileNetV2 as mob_org
from torchvision.models.Setting1_LWRN import MobileNetLWRF as setting1
from torchvision.models.Setting2_LWRN import MobileNetLWRF as setting2
from torchvision.models.Setting3_LWRN import MobileNetLWRF as setting3
from torchvision.models.Vanila_MNV2_4BN import MobileNetV2 as MNV2_4BN
from torchvision.models.Vanila_MNV2_7BN import MobileNetV2 as MNV2_7BN
from torchvision.models.Setting1_backbone import MobileNetV2 as S1_BB

pt_models = { 'resnet18': models.resnet18, 'resnet50': models.resnet50,
              'alexnet': models.alexnet, 'CMTL': CMTL, 'SANet': SANet,
              'vgg16': models.vgg16, 'squeezenet': models.squeezenet1_0,
              'densenet': models.densenet161, 'MobileNetV2': MobileNetV2,
              'MCNN': MCNN, 'VGG': VGG, 'VGG_decoder': VGG_decoder, 'FPN': FPN,
              'ResNetLW': ResNetLW, 'DeepLabv3': DeepLabv3, 'MobileNetLWRF': MobileNetLWRF,
              'ShuffleNetLWRF': ShuffleNetLWRF, 'CSRNet': CSRNet, 'Res50': Res50,
              'mob_org': mob_org, 'setting1': setting1, 'setting2': setting2, 'setting3': setting3,
              'MNV2_4BN': MNV2_4BN, 'MNV2_7BN': MNV2_7BN, 'S1_BB': S1_BB
              }"""
pt_models = {'vgg16': vgg16, 'vgg19': vgg19, 'vgg11': vgg11}

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './DULR-display-save-mat'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)


def main():
    test()
   

def test():
    # net = MCNN()

    net = pt_models['vgg19'](pretrained=True)

    flops, params = get_model_complexity_info(net, (1920, 1080), as_strings=False, print_per_layer_stat=False)
    print('FLOPs')
    print(flops)
    print('Params')
    print(params)



if __name__ == '__main__':
    main()




