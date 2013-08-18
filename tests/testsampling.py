from dtcwt.sampling import scale

import numpy as np

def test_scale_lanczos():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = scale(X, (300, 210), 'lanczos')
    assert Xrs.shape == (300, 210)

    # And down
    Xrecon = scale(Xrs, X.shape, 'lanczos')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing to within 5%?
    assert np.all(np.abs(X-Xrecon) < 5e-2)

def test_scale_bilinear():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = scale(X, (300, 210), 'bilinear')
    assert Xrs.shape == (300, 210)

    # And down
    Xrecon = scale(Xrs, X.shape, 'bilinear')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing to within 30% (bilinear sucks)
    assert np.all(np.abs(X-Xrecon) < 3e-1)

def test_scale_nearest():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = scale(X, (200, 240), 'nearest')
    assert Xrs.shape == (200, 240)

    # And down
    Xrecon = scale(Xrs, X.shape, 'nearest')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing to within 1% (nearest neighbour should be exact)
    assert np.all(np.abs(X-Xrecon) < 1e-2)
