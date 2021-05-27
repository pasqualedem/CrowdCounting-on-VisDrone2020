from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.CC import CrowdCounter
from dataset.visdrone import load_test, load_train_val, cfg_data
from train import Trainer
from config import cfg
import numpy as np
import torch


def load_CC():
    cc = CrowdCounter([0], 'MobileCount')
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
    res = evaluate_model(model_function=load_CC,
                         data_function=load_test,
                         bs=2,
                         n_workers=2,
                         losses={'mse': mean_squared_error, 'mae': mean_absolute_error},
                         )
    print(res)


def train_net():
    trainer = Trainer(dataloader=prova,
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
