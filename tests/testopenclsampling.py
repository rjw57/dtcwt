import dtcwt
import dtcwt.opencl.sampling
from dtcwt.sampling import rescale

from .util import skip_if_no_cl

import numpy as np

@skip_if_no_cl
def setup():
    dtcwt.push_backend('opencl')

@skip_if_no_cl
def teardown():
    dtcwt.pop_backend()

@skip_if_no_cl
def test_correct_backend():
    # Test that the correct backend will be used for sampling
    assert dtcwt._sampling is dtcwt.opencl.sampling

@skip_if_no_cl
def test_rescale_nearest():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = rescale(X, (200, 240), 'nearest')
    assert Xrs.shape == (200, 240)

    # And down
    Xrecon = rescale(Xrs, X.shape, 'nearest')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing
    assert np.all(np.abs(X-Xrecon) < 1e-8)

@skip_if_no_cl
def test_rescale_pixel_centre():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = rescale(X, (200, 240), 'nearest')
    assert Xrs.shape == (200, 240)

    # Output should be 4x4 blocks identical to original
    for dx, dy in ((0,0), (0,1), (1,0), (1,1)):
        Y = Xrs[dx::2,dy::2]
        assert np.all(np.abs(X-Y) < 1e-8)
