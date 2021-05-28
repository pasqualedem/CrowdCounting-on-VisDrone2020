"""
Script to compute latency and fps of a model
"""
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
from config import cfg
from models.CC import CrowdCounter
from dataset.random import RandomDataset

def load_CC():
    cc = CrowdCounter([0], 'MobileCount')
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc


models = {'vgg16': vgg16,
          'vgg19': vgg19,
          'vgg11': vgg11,
          'MobileCount': MobileCount,
          'MobileCountx1_25': MobileCountx1_25,
          'MobileCountx2': MobileCountx2,
          'CC': load_CC
          }


def measure_forward(model, x):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_pred = model.predict(x.cuda())
    torch.cuda.synchronize()
    elapsed_fp = time.perf_counter() - t0

    return elapsed_fp


def measure_fps(model, dataset):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for x in dataset:
            _ = model.predict(x.cuda())
    torch.cuda.synchronize()
    elapsed_fp = time.perf_counter() - t0

    return elapsed_fp


def benchmark_forward(model, x, num_runs, num_warmup_runs):

    print('\nStarting warmup')
    # DRY RUNS
    for i in tqdm(range(num_warmup_runs)):
        _ = measure_forward(model, x)

    print('\nDone, now benchmarking')

    # START BENCHMARKING
    t_forward = []
    for x in tqdm(num_runs):
        t_fp = measure_forward(model, x)
        t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


def benchmark_fps(model, dataset, num_runs, num_warmup_runs):

    print('\nStarting warmup')
    # DRY RUNS
    x = None
    for it in dataset:
        x = it
        break

    for i in tqdm(range(num_warmup_runs)):
        _ = measure_forward(model, x)

    print('\nDone, now benchmarking')

    # START BENCHMARKING
    t_forward = []
    for i in tqdm(range(num_runs)):
        t_fp = measure_fps(model, dataset)
        t_forward.append(t_fp)

    # free memory
    del model

    return t_forward


class Benchmarker:
    def __init__(self, model, dataset, batch_sizes, out_file, n_workers):
        torch.manual_seed(1234)
        cudnn.benchmark = True
        # transfer the model on GPU
        self.model = model.cuda().eval()
        self.dataset = dataset
        self.out_file = out_file
        self.batch_sizes = batch_sizes
        self.n_workers = n_workers

    def bench_forward(self, num_runs, num_warmup_runs):

        mean_tfp = []
        std_tfp = []
        for i, bs in enumerate(self.batch_sizes):
            print('\nBatch size is: ' + str(bs))
            print('--------------------------')
            data_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=bs, shuffle=False, num_workers=self.n_workers
            )
            tmp = benchmark_forward(model, num_runs, num_warmup_runs, data_loader)
            # NOTE: we are estimating inference time per image
            mean_tfp.append(np.asarray(tmp).mean() / bs * 1e3)
            std_tfp.append(np.asarray(tmp).std() / bs * 1e3)

        self.out_results({'mean (ms)': mean_tfp, 'std (ms)': std_tfp})

        # force garbage collection
        gc.collect()

    def bench_fps(self, num_runs, num_warmup_runs):

        mean_fps = []
        std_fps = []
        for i, bs in enumerate(self.batch_sizes):
            print('\nBatch size is: ' + str(bs))
            print('--------------------------')
            data_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=bs, shuffle=False, num_workers=self.n_workers
            )
            tmp = benchmark_fps(model, data_loader, num_runs, num_warmup_runs)
            mean_fps.append((1 / (np.asarray(tmp) / len(self.dataset))).mean())
            std_fps.append((1 / (np.asarray(tmp) / len(self.dataset))).std())

        self.out_results({'mean (fps)': mean_fps, 'std (fps)': std_fps})

        # force garbage collection
        gc.collect()

    def out_results(self, dictionary):
        df = pd.DataFrame(dictionary, index=self.batch_sizes)
        size = 'Input size: ' + str(self.dataset.shape())
        table = tabulate(df, headers='keys', tablefmt='psql')
        device = 'Device: ' + torch.cuda.get_device_name(torch.cuda.current_device())
        result = "%s\n%s\n%s" % (device, size, table)

        print(result)
        if self.out_file != 'none':
            with open(self.out_file, 'w') as f:
                f.write(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute time for an image.')
    parser.add_argument('model', type=str, help='model name to be evaluated')
    parser.add_argument('mode', type=str, help='can be forward or fps ')
    parser.add_argument('--input_size', type=str, default="(3, 512, 512)",
                        help='size of the input image size. default is 3x512x512')
    parser.add_argument('--batch_sizes', type=str, default="[1, 2, 4]",
                        help='list of batch size to try. default is [1, 2, 4]')
    parser.add_argument('--n_workers', type=int, default="2",
                        help='number of workers for multiprocessing. default is 2')
    parser.add_argument('--num_runs', type=int, default=105,
                        help='number of runs to compute average forward timing. default is 105')
    parser.add_argument('--num_warmup_runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')
    parser.add_argument('--dataset_len', type=int, default=100,
                        help='size of the dataset')
    parser.add_argument('--file', type=str, default='none',
                        help='where to save the file. default is none')
    args = parser.parse_args()

    model = models[args.model]()
    bs = literal_eval(args.batch_sizes)
    in_size = literal_eval(args.input_size)

    dataset = RandomDataset(in_size, args.dataset_len)

    print('Model is loaded, start forwarding.')
    benchmarker = Benchmarker(model, dataset, bs, args.file, args.n_workers)
    if args.mode == 'forward':
        benchmarker.bench_forward(args.num_warmup_runs, args.num_runs)
    elif args.mode == 'fps':
        benchmarker.bench_fps(args.num_warmup_runs, args.num_runs)
    else:
        print('Wrong mode given: ' + str(args.mode))
