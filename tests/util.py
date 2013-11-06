import numpy as np

TOLERANCE = 1e-12

def assert_almost_equal(a, b, tolerance=TOLERANCE):
    md = np.abs(a-b).max()
    if md <= tolerance:
        return

    raise AssertionError(
            'Arrays differ by a maximum of {0} which is greater than the tolerance of {1}'.
            format(md, tolerance))
