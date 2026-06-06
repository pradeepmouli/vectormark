# SPDX-License-Identifier: MIT
import numpy as np


def test_mean_delta_e_zero_for_identical_and_positive_for_different():
    from vectormark.color import mean_delta_e
    a = np.full((4, 4, 3), 120, np.uint8)
    assert mean_delta_e(a, a) == 0.0
    b = a.copy()
    b[:, :, 0] = 200          # shift red channel
    assert mean_delta_e(a, b) > 0.02
