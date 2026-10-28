"""The purged splitter must never let a training label overlap the test block."""

import numpy as np
import pytest

from vrp.models import PurgedTimeSeriesSplit, huber_loss, make_huber_objective


@pytest.mark.parametrize("n", [600, 1990])
def test_purge_and_embargo_gap(n):
    horizon, embargo = 21, 5
    splits = list(PurgedTimeSeriesSplit(5, horizon, embargo).split(n))
    assert len(splits) == 5
    for train, test in splits:
        assert train.max() + horizon + embargo < test.min()  # labels end before test starts
        assert np.all(np.diff(test) == 1)
    assert splits[-1][1].max() == n - 1                     # last fold reaches the end


def test_huber_gradient_matches_loss():
    y, p, delta = np.array([0.0, 0.0]), np.array([0.01, 0.5]), 0.02
    grad, hess = make_huber_objective(delta)(y, p)
    assert np.allclose(grad, [0.01, delta]) and np.all(hess == 1)
    assert huber_loss(y, p, delta) == pytest.approx((0.5 * 0.01**2 + delta * (0.5 - delta / 2)) / 2)
