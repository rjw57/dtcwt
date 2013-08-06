import os

import numpy as np
from dtcwt import dtwavexfm2

def setup():
    global lena
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

def test_simple():
    Yl, Yh = dtwavexfm2(lena)

def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena, include_scale=True)

def test_0_levels():
    Yl, Yh = dtwavexfm2(lena, nlevels=0)
    assert np.all(np.abs(Yl - lena) < 1e-5)
    assert len(Yh) == 0

def test_0_levels_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena, nlevels=0, include_scale=True)
    assert np.all(np.abs(Yl - lena) < 1e-5)
    assert len(Yh) == 0
    assert len(Yscale) == 0

# vim:sw=4:sts=4:et
