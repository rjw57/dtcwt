import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm2, dtwaveifm2, biort, qshift

TOLERANCE = 1e-12

def setup():
    global lena, lena_crop
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']
    lena_crop = lena[:233, :301]

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

@attr('transform')
def test_reconstruct():
    # Reconstruction up to tolerance
    Yl, Yh = dtwavexfm2(lena)
    lena_recon = dtwaveifm2(Yl, Yh)
    assert np.all(np.abs(lena_recon - lena) < TOLERANCE)

@attr('transform')
def test_reconstruct_crop():
    # Reconstruction up to tolerance
    Yl_crop, Yh_crop = dtwavexfm2(lena_crop)
    lena_recon = dtwaveifm2(Yl_crop, Yh_crop)[:lena_crop.shape[0], :lena_crop.shape[1]]
    assert np.all(np.abs(lena_recon - lena_crop) < TOLERANCE)

@attr('transform')
def test_reconstruct_custom_filter():
    # Reconstruction up to tolerance
    Yl, Yh = dtwavexfm2(lena, 4, biort('legall'), qshift('qshift_06'))
    lena_recon = dtwaveifm2(Yl, Yh, biort('legall'), qshift('qshift_06'))
    assert np.all(np.abs(lena_recon - lena) < TOLERANCE)


# vim:sw=4:sts=4:et
