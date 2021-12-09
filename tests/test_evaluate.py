import os
from tempfile import NamedTemporaryFile

import torch.cuda

from models.CC import CrowdCounter
from config import cfg
from dataset.visdrone import cfg_data, make_dataframe, VisDroneDataset
from sklearn.metrics import mean_absolute_error

import evaluate as eval
import pytest
import json


def load_test_subset():
    """
    Create a VisDroneDataset object in test mode with 3 batches
    @return: the visdrone testset
    """
    df = make_dataframe(os.path.join(cfg_data.DATA_PATH, cfg_data.DATA_SUBFOLDER, 'test'), cfg_data.SIZE)
    df = df[0:cfg.VAL_BATCH_SIZE * 3]
    ds = VisDroneDataset(df, train=cfg.TRAIN, gt_transform=cfg.GT_TRANSFORM)
    return ds


@pytest.mark.evaluate
def test_metric_saver():
    metric_file = NamedTemporaryFile()
    metric_file.close()
    metrics = {'mse': 23.2, 'mae': 12.3}
    eval.metrics_saver(metric_file.name, metrics)
    with open(metric_file.name, 'r') as m:
        red_metrics = json.load(m)
    assert red_metrics == metrics


class TestPreliminarTest:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.eval_params, self.global_params = eval.load_yaml_eval_params()
        eval.initialize_eval_params(self.eval_params, self.global_params)

    @pytest.mark.evaluate
    def test_param_load(self):
        assert issubclass(type(self.eval_params), dict)
        assert issubclass(type(self.global_params), dict)

    @pytest.mark.evaluate
    def test_initialize_eval_params(self):
        assert cfg.GPU
        assert cfg_data.DATA_PATH

    @pytest.mark.evaluate
    def test_load_CC_test(self):
        cfg.PRE_TRAINED = None
        model = eval.load_CC_test()
        assert type(model) == CrowdCounter

    @pytest.mark.evaluate
    @pytest.mark.slow
    def test_evaluate(self):
        losses = {'mae': mean_absolute_error}
        cfg.PRE_TRAINED = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.GPU = device
        metrics = eval.evaluate_model(eval.load_CC_test, load_test_subset, cfg.TEST_BATCH_SIZE, cfg.N_WORKERS, losses,
                                      device,
                                      None)
        assert len(metrics) == 1
        assert metrics.get('mae') >= 0
