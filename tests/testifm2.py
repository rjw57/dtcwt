import os
from nose.tools import raises

import numpy as np
from dtcwt import dtwavexfm2, dtwaveifm2

def setup():
    global lena, lena_crop, Yl, Yh, Yl_crop, Yh_crop
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']
    lena_crop = lena[:233, :301]
    Yl, Yh = dtwavexfm2(lena)
    Yl_crop, Yh_crop = dtwavexfm2(lena_crop)

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

def test_reconstruct():
    # Reconstruction up to tolerance
    lena_recon = dtwaveifm2(Yl, Yh)
    assert np.all(np.abs(lena_recon - lena) < 1e-3)

def test_reconstruct_ctop():
    # Reconstruction up to tolerance
    lena_recon = dtwaveifm2(Yl_crop, Yh_crop)[:lena_crop.shape[0], :lena_crop.shape[1]]
    assert np.all(np.abs(lena_recon - lena_crop) < 1e-3)

# vim:sw=4:sts=4:et
