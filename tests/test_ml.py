import json
import pytest
import os

MIN_METRICS = {
    'rmse': 45.0,
    'mae': 35.0}

METRICS_FILE = 'metrics.json'
GIT_METRICS_FILE = 'https://raw.githubusercontent.com/se4ai2122-cs-uniba/Drone-CrowdCounting/main/metrics.json'
DEPLOYED_METRICS_FILE = 'deployed_metrics.json'


@pytest.mark.ml
def test_metrics():
    with open(METRICS_FILE) as metrics_file:
        metrics = json.load(metrics_file)

    for metric in MIN_METRICS:
        assert MIN_METRICS[metric] > metrics[metric]

    os.system('wget {} -O {}'.format(GIT_METRICS_FILE, DEPLOYED_METRICS_FILE))
    print('Going to test')
    if os.path.exists(DEPLOYED_METRICS_FILE):
        print('Testing new metrics')
        with open(DEPLOYED_METRICS_FILE) as metrics_file:
            deployed_metrics = json.load(metrics_file)

        for metric in MIN_METRICS:
            print("new: {} old: {}".format(metrics[metric], deployed_metrics[metric]))
            assert metrics[metric] <= deployed_metrics[metric]

