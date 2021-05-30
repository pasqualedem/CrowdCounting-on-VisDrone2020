import numpy as np
import os
import time

import torch
from torch import nn

from tensorboardX import SummaryWriter

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given number of consecutive epochs"""
    def __init__(self, patience=1, delta=1e-4):
        """
        Instantiate an EarlyStopping object.

        :param patience: The number of consecutive epochs to wait.
        :param delta: The minimum change of the monitored quantity.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.should_stop = False
        self.counter = 0
        self.best_state = None

    def should_stop(self):
        return self.should_stop

    def __call__(self, loss):
        """
        Call the object.

        :param loss: The validation loss measured.
        """
        # Check if an improved of the loss happened
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        # Check if the training should stop
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print(m)


def weights_normal_init(*models):
    for model in models:
        dev = 0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def logger(exp_path, exp_name):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path + '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'

    cfg_file = open('./config.py', "r")
    cfg_lines = cfg_file.readlines()

    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    return writer, log_file


def logger_txt(log_file, epoch, scores):
    mae, mse, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)

    with open(log_file, 'a') as f:
        f.write('=' * 15 + '+' * 15 + '=' * 15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('    [mae %.2f mse %.2f], [val loss %.4f]\n' % (mae, mse, loss))
        f.write('=' * 15 + '+' * 15 + '=' * 15 + '\n\n')


def print_summary(epoch, scores, train_record, for_time, train_time, val_time):
    mae, mse, loss = scores
    print('Epoch ' + str(epoch) + ' | ', end='')
    print('    [mae %.2f mse %.2f], [val loss %.4f] [forward time %.2f] [train/valid time %.2f / %.2f] --- '
          % (mae, mse, loss, for_time, train_time, val_time),
          end='')
    print('[best] [model: %s] , [mae %.2f], [mse %.2f]' % (train_record['best_model_name'],
                                                           train_record['best_mae'],
                                                           train_record['best_rmse']))


def update_model(state_dict, epoch, exp_path, exp_name, scores, train_record, log_file):
    mae, rmse, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_rmse_%.1f' % (epoch + 1, mae, rmse)
    logger_txt(log_file, epoch, scores)

    if mae < train_record['best_mae'] or rmse < train_record['best_rmse']:
        train_record['best_model_name'] = snapshot_name
        torch.save(state_dict, os.path.join(exp_path, exp_name, snapshot_name + '.pth'))

    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
    if rmse < train_record['best_rmse']:
        train_record['best_rmse'] = rmse

    return train_record


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        if hasattr(cur_val, '__iter__'):
            for val in cur_val:
                self._update(val)
        else:
            self._update(cur_val)

    def _update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, cur_val, class_id):
        self.cur_val[class_id] = cur_val
        self.sum[class_id] += cur_val
        self.count[class_id] += 1
        self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff








