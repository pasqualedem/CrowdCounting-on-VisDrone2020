import pytest

from utils import EarlyStopping


@pytest.mark.utils
def test_EarlyStopping():

    delta = 1e-4
    patience = 5

    early_stop = EarlyStopping(patience, delta)
    losses = [2, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    for i in range(len(losses)):
        early_stop(losses[i])
        if early_stop.should_stop:
            break
    assert i == 7
