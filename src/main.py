import argparse

from ast import literal_eval
import matplotlib.pyplot as plt
from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.CC import CrowdCounter
from dataset.visdrone import load_test, load_train_val, cfg_data
from dataset.run_datasets import make_dataset
from run import run_model, run_transforms
from train import Trainer
from config import cfg
import numpy as np
import torch


def load_CC_train():
    cc = CrowdCounter([0], cfg.NET)
    return cc


def load_CC_test():
    cc = CrowdCounter([0], cfg.NET)
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc


class ProvaSet(torch.utils.data.Dataset):

    def __getitem__(self, item):
        return torch.rand(3, 50, 50), torch.rand(50, 50)

    def __len__(self):
        return 50


def prova():
    train_set = ProvaSet()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.N_WORKERS, shuffle=True)

    val_set = ProvaSet()
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.VAL_BATCH_SIZE, num_workers=cfg.N_WORKERS, shuffle=True)

    return train_loader, val_loader


def test_net():
    res = evaluate_model(model_function=load_CC_test,
                         data_function=load_test,
                         bs=cfg.TEST_BATCH_SIZE,
                         n_workers=cfg.N_WORKERS,
                         losses={'rmse': lambda x, y: mean_squared_error(x, y, squared=False),
                                 'mae': mean_absolute_error},
                         out_prediction=None
                         )
    print(res)


def run_net(in_file, callbacks):
    dataset = make_dataset(in_file)

    transforms = run_transforms(cfg_data.MEAN, cfg_data.STD, cfg_data.SIZE)
    dataset.set_transforms(transforms)

    def count_callback(input, prediction, other):
        print(other + ' Count: ' + str(torch.sum(prediction.squeeze()).item() / cfg_data.LOG_PARA))

    def save_callback(input, prediction, other):
        plt.imsave(other + '.png', prediction.squeeze(), cmap='jet')

    call_dict = {'save_callback': save_callback, 'count_callback': count_callback}

    callbacks = [(call_dict[call] if type(call) == 'str' else call) for call in callbacks]

    run_model(load_CC_test, dataset, cfg.TEST_BATCH_SIZE, cfg.N_WORKERS, callbacks)


def train_net():
    trainer = Trainer(dataloader=load_train_val,
                      cfg_data=cfg_data,
                      net_fun=load_CC_train
                      )
    trainer.train()


if __name__ == '__main__':
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser(description='Execute a training, an evaluation or run the net on some example')
    parser.add_argument('mode', type=str, help='can be train, test or run')
    parser.add_argument('--in_file', type=str, help='in run mode, the input file or folder to be processed')
    parser.add_argument('--callbacks', type=str,
                        help='List of callbacks, they can be [\'save_callback\', \'count_callback\']')
    args = parser.parse_args()

    callbacks = literal_eval(args.callbacks)
    if args.mode == 'train':
        train_net()
    elif args.mode == 'test':
        test_net()
    elif args.mode == 'run':
        run_net(args.in_file, callbacks)
