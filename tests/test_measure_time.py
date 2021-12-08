import pytest
from dataset.stochastic import RandomDataset
from measure_time import load_CC05, Benchmarker
from multiprocessing import cpu_count
from config import cfg

@pytest.mark.time
@pytest.mark.cuda
@pytest.mark.slow
def test_time():
    size = (3, 128, 256)
    dataset_len = 50
    dataset = RandomDataset(size, dataset_len)
    workers = cpu_count()
    batch_sizes = [1, 2]
    file = 'none'
    cfg.PRE_TRAINED = None
    model = load_CC05()
    num_runs = 2
    num_warmup_runs = 1

    benchmarker = Benchmarker(model, dataset, batch_sizes, file, workers)
    benchmarker.bench_forward(num_runs, num_warmup_runs)
    benchmarker.bench_fps(num_runs, num_warmup_runs)