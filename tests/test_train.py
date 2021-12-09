import os

import sklearn
import torch

from train import load_CC_train, load_yaml_train_params, initialize_dynamic_params, Trainer
from dataset.visdrone import load_train_val, cfg_data, make_dataframe, VisDroneDataset
from config import cfg
from models.CC import CrowdCounter
from easydict import EasyDict
import tempfile

import pytest
import torch.utils.data as data_utils


def load_train_subset():
    """
     Create a train and validation DataLoader from the specified folder that is a subset with 3 batches
     config values are used (VAL_SIZE, VAL_BATCH_SIZE, N_WORKERS)

     @return: the train and validation DataLoader
     """
    df = make_dataframe(os.path.join(cfg_data.DATA_PATH, cfg_data.DATA_SUBFOLDER, 'train'), cfg_data.SIZE)
    # Split the dataframe in train and validation
    train_df, valid_df = sklearn.model_selection.train_test_split(
        df, test_size=cfg.VAL_SIZE, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_df = train_df[0:cfg.TRAIN_BATCH_SIZE * 3]
    valid_df = valid_df[0:cfg.VAL_BATCH_SIZE * 3]

    train_set = VisDroneDataset(train_df)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.TRAIN_BATCH_SIZE, shuffle=True)

    val_set = VisDroneDataset(valid_df)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.VAL_BATCH_SIZE, num_workers=cfg.VAL_BATCH_SIZE, shuffle=True)
    return train_loader, val_loader


class TestPreliminarTrain:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.train_params, self.global_params = load_yaml_train_params()
        cfg.update(self.train_params)
        cfg_data.update(self.global_params)
        initialize_dynamic_params()

    @pytest.mark.train
    def test_param_load(self):
        assert issubclass(type(self.train_params), dict)
        assert issubclass(type(self.global_params), dict)

    @pytest.mark.train
    def test_param_init(self):
        assert cfg.GPU
        assert cfg.PRETRAINED
        assert type(cfg.OPTIM[0]) == str
        assert type(cfg.OPTIM[1]) == EasyDict

    @pytest.mark.train
    def test_load_CC_train(self):
        model = load_CC_train()
        assert type(model) == CrowdCounter

    @pytest.mark.train
    @pytest.mark.slow
    @pytest.mark.cuda
    def test_train(self):
        cfg.update(self.train_params)
        cfg_data.update(self.global_params)
        initialize_dynamic_params()
        cfg.PRETRAINED = None
        cfg.MAX_EPOCH = 1
        cfg.EXP_PATH = tempfile.mkdtemp()
        cfg.PRINT_FREQ = 1

        trainer = Trainer(dataloader=load_train_subset,
                          cfg_data=cfg_data,
                          net_fun=load_CC_train
                          )
        trainer.train()
        assert os.path.exists(os.path.join(cfg.EXP_PATH, 'model.pth'))
        assert os.path.exists(os.path.join(cfg.EXP_PATH, 'log.txt'))