from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.CC import CrowdCounter
from dataset.visdrone import load_test, load_train_val, cfg_data
from train import Trainer
from config import cfg
import numpy as np
import torch


def load_CC():
    cc = CrowdCounter([0], 'MobileCountx2')
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc


def test_net():
    res = evaluate_model(model_function=load_CC,
                         data_function=load_test,
                         bs=8,
                         n_workers=2,
                         losses={'mse': mean_squared_error, 'mae': mean_absolute_error},
                         )
    print(res)


def train_net():
    trainer = Trainer(dataloader=load_train_val,
                      cfg_data=cfg_data,
                      net_fun=load_CC
                      )
    trainer.train()


if __name__ == '__main__':
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    train_net()
