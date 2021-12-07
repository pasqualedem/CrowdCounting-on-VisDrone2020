from train import load_CC_train, load_yaml_train_params, initialize_dynamic_params
from config import cfg
from models.CC import CrowdCounter
from dataset.visdrone import cfg_data
from easydict import EasyDict

import pytest


class TestPreliminarTrain:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.train_params, self.global_params = load_yaml_train_params()
        cfg.update(self.train_params)
        cfg_data.update(self.global_params)
        initialize_dynamic_params()

    @pytest.mark.training
    def test_param_load(self):
        assert issubclass(type(self.train_params), dict)
        assert issubclass(type(self.global_params), dict)

    @pytest.mark.training
    def test_param_init(self):
        assert cfg.GPU
        assert cfg.PRETRAINED
        assert type(cfg.OPTIM[0]) == str
        assert type(cfg.OPTIM[1]) == EasyDict

    @pytest.mark.training
    def test_load_CC_train(self):
        model = load_CC_train()
        assert type(model) == CrowdCounter
