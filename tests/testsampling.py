from dtcwt.sampling import rescale

import numpy as np

def test_rescale_lanczos():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = rescale(X, (300, 210), 'lanczos')
    assert Xrs.shape == (300, 210)

    # And down
    Xrecon = rescale(Xrs, X.shape, 'lanczos')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing to within 5%?
    assert np.all(np.abs(X-Xrecon) < 5e-2)

def test_rescale_bilinear():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = rescale(X, (300, 210), 'bilinear')
    assert Xrs.shape == (300, 210)

    # And down
    Xrecon = rescale(Xrs, X.shape, 'bilinear')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing to within 30% (bilinear sucks)
    assert np.all(np.abs(X-Xrecon) < 3e-1)

def test_rescale_nearest():
    # Create random 100x120 image
    X = np.random.rand(100,120)

    # Re size up
    Xrs = rescale(X, (200, 240), 'nearest')
    assert Xrs.shape == (200, 240)

    # And down
    Xrecon = rescale(Xrs, X.shape, 'nearest')
    assert Xrecon.shape == X.shape

    # Got back roughly the same thing to within 1% (nearest neighbour should be exact)
    assert np.all(np.abs(X-Xrecon) < 1e-2)

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
