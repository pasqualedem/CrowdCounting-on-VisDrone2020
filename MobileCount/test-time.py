import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from PIL import Image, ImageOps
import torchvision

import time
import argparse
import gc

import datetime
import os
from torchvision.models import vgg16, vgg19, vgg11
from model_flows import flows

pt_models = {'vgg16': vgg16, 'vgg19': vgg19, 'vgg11': vgg11, 'flows': flows}


def measure(model, x):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    return elapsed_fp


def benchmark(model, x):
    # transfer the model on GPU
    model = model.cuda().eval()

    # DRY RUNS
    for i in range(10):
        _ = measure(model, x)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []
    for i in range(10):
        t_fp = measure(model, x)
        t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


def main():
    # fix random
    torch.manual_seed(1234)

    # create model
    # print("=> creating model '{}'".format(m))
    model = pt_models['flows']()

    cudnn.benchmark = True

    scale = 0.875

    # print('Images transformed from size {} to {}'.format(
    #     int(round(max(model.input_size) / scale)),
    #     model.input_size))

    # mean_tfp = []
    # std_tfp = []
    # x = torch.randn(1, 3, 224, 224).cuda()
    # tmp = benchmark(model, x)
    # # NOTE: we are estimating inference time per image
    # mean_tfp.append(np.asarray(tmp).mean() / 1 * 1e3)
    # std_tfp.append(np.asarray(tmp).std() / 1 * 1e3)
    #
    # print(mean_tfp, std_tfp)

    batch_sizes = [1, 2, 4]
    mean_tfp = []
    std_tfp = []
    for i, bs in enumerate(batch_sizes):
        # x = torch.randn(bs, 3, 224, 224).cuda()
        x = torch.randn(bs, 3, 1920, 1080).cuda()
        tmp = benchmark(model, x)
        # NOTE: we are estimating inference time per image
        mean_tfp.append(np.asarray(tmp).mean() / bs * 1e3)
        std_tfp.append(np.asarray(tmp).std() / bs * 1e3)
        print(np.asarray(tmp).mean() / bs * 1e3)

    print(mean_tfp, std_tfp)

    # force garbage collection
    gc.collect()


if __name__ == '__main__':
    main()
