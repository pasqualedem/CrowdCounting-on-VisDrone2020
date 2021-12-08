import pytest
from dataset.calculate_mean_std import calculate


@pytest.mark.slow
def test_calculate():
    mean, std = calculate()
    assert (mean >= 0).all()
    assert (std >= 0).all()
