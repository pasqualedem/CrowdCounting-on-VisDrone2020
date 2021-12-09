import json
import pytest

MIN_METRICS = {
    'rmse': 45.0,
    'mae': 35.0}

@pytest.mark.ml
def test_metrics():
    with open('metrics.json') as metrics_file:
        metrics = json.load(metrics_file)

    for metric in MIN_METRICS:
        assert MIN_METRICS[metric] > metrics[metric]
