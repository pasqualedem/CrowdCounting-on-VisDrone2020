import argparse

from ast import literal_eval
from callbacks import call_dict
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
    """
    Load CrowdCounter model net for training mode
    """
    cc = CrowdCounter([0], cfg.NET)
    return cc


def load_CC_test():

    """
    Load CrowdCounter model net for testing mode
    """
    cc = CrowdCounter([0], cfg.NET)
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc


def test_net():
    """
    Test a model on a specific test set
    Must specify the function tha returns the model and the dataset
    """
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
    """
    Run the model on a given file or folder

    @param in_file: media file or folder of images
    @param callbacks: list of callbacks to be called after every forward operation
    """
    dataset = make_dataset(in_file)

    transforms = run_transforms(cfg_data.MEAN, cfg_data.STD, cfg_data.SIZE)
    dataset.set_transforms(transforms)

    callbacks_list = [(call_dict[call] if type(call) == str else call) for call in callbacks]

    run_model(load_CC_test, dataset, cfg.TEST_BATCH_SIZE, cfg.N_WORKERS, callbacks_list)


def train_net():
    """
    Train the given model on a given data loader
    """
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
    parser.add_argument('--path', type=str, help='in run mode, the input file or folder to be processed')
    parser.add_argument('--callbacks', type=str,
                        help='List of callbacks, they can be [\'save_callback\', \'count_callback\']')
    args = parser.parse_args()

    if args.callbacks is not None:
        callbacks = literal_eval(args.callbacks)
    else:
        callbacks = []
    if args.mode == 'train':
        train_net()
    elif args.mode == 'test':
        test_net()
    elif args.mode == 'run':
        run_net(args.path, callbacks)
