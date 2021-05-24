"""
Script to compute latency and fps of a model
"""
import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import gc

import torch
import torch.backends.cudnn as cudnn
from torchvision.models import vgg16, vgg19, vgg11
from models.MobileCount import MobileCount
from models.MobileCountx1_25 import MobileCount as MobileCountx1_25
from models.MobileCountx2 import MobileCount as MobileCountx2
from tabulate import tabulate

models = {'vgg16': vgg16,
          'vgg19': vgg19,
          'vgg11': vgg11,
          'MobileCount': MobileCount,
          'MobileCountx1_25': MobileCountx1_25,
          'MobileCountx2': MobileCountx2
          }


def measure(model, x):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    return elapsed_fp


def benchmark(model, x, num_runs, num_warmup_runs):
    # transfer the model on GPU
    model = model.cuda().eval()

    print('\nStarting warmup')
    # DRY RUNS
    for i in tqdm(range(num_warmup_runs)):
        _ = measure(model, x)

    print('\nDone, now benchmarking')

    # START BENCHMARKING
    t_forward = []
    t_backward = []
    for i in tqdm(range(num_runs)):
        t_fp = measure(model, x)
        t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


def test(model, dim, batch_sizes, num_runs, num_warmup_runs, file):
    # fix random
    torch.manual_seed(1234)

    cudnn.benchmark = True

    scale = 0.875

    mean_tfp = []
    std_tfp = []
    for i, bs in enumerate(batch_sizes):
        print('\nBatch size is: ' + str(bs))
        print('--------------------------')
        x = torch.randn(bs, *dim).cuda()
        tmp = benchmark(model, x, num_runs, num_warmup_runs)
        # NOTE: we are estimating inference time per image
        mean_tfp.append(np.asarray(tmp).mean() / bs * 1e3)
        std_tfp.append(np.asarray(tmp).std() / bs * 1e3)


    df = pd.DataFrame({'mean (ms)': mean_tfp, 'std (ms)': std_tfp}, index=batch_sizes)
    size = 'Input size: ' + str(dim)
    table = tabulate(df, headers='keys', tablefmt='psql')
    device = 'Device: ' + torch.cuda.get_device_name(torch.cuda.current_device())
    result = "%s\n%s\n%s" % (device, size, table)

    print(result)
    if file != 'none':
        with open(file, 'w') as f:
            f.write(result)

    # force garbage collection
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute time for an image.')
    parser.add_argument('model', type=str, help='model name to be evaluated')
    parser.add_argument('--input_size', type=str, default="(3, 512, 512)",
                        help='size of the input image size. default is 3x512x512')
    parser.add_argument('--batch_sizes', type=str, default="[1, 2, 4]",
                        help='list of batch size to try. default is [1, 2, 4]')
    parser.add_argument('--num_runs', type=int, default=105,
                        help='number of runs to compute average forward timing. default is 105')
    parser.add_argument('--num_warmup_runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')
    parser.add_argument('--file', type=str, default='none',
                        help='where to save the file. default is none')
    args = parser.parse_args()

    model = models[args.model]()
    print('Model is loaded, start forwarding.')
    test(model, literal_eval(args.input_size), literal_eval(args.batch_sizes), args.num_runs, args.num_warmup_runs, args.file)