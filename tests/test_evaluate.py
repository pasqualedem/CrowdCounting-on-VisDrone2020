from tempfile import NamedTemporaryFile
from models.CC import CrowdCounter
from config import cfg
from dataset.visdrone import cfg_data

import evaluate as eval
import pytest
import json


@pytest.mark.testing
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

    @pytest.mark.testing
    def test_param_load(self):
        assert issubclass(type(self.eval_params), dict)
        assert issubclass(type(self.global_params), dict)

    @pytest.mark.testing
    def test_initialize_eval_params(self):
        assert cfg.GPU
        assert cfg_data.DATA_PATH

    @pytest.mark.testing
    def test_load_CC_test(self):
        cfg.PRE_TRAINED = None
        model = eval.load_CC_test()
        assert type(model) == CrowdCounter