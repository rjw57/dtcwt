import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm2, dtwaveifm2, biort, qshift
from dtcwt.lowlevel import coldfilt

TOLERANCE = 1e-12

def setup():
    global lena
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

    global verif
    verif = np.load(os.path.join(os.path.dirname(__file__), 'verification.npz'))

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

def test_lena_loaded():
    assert verif is not None
    assert 'lena_coldfilt' in verif

def test_coldfilt():
    h0o, g0o, h1o, g1o = biort('near_sym_b')
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    A = coldfilt(lena, h1b, h1a)
    assert np.abs(A - verif['lena_coldfilt']).max() < TOLERANCE

def test_dtwavexfm2():
    Yl, Yh, Yscale = dtwavexfm2(lena, 4, 'near_sym_a', 'qshift_a', include_scale=True)
    assert np.abs(Yl - verif['lena_Yl']).max() < TOLERANCE

    assert len(Yh) == verif['lena_Yh'].shape[0]
    for a, b in zip(Yh, verif['lena_Yh']):
        assert np.abs(a-b).max() < TOLERANCE

    assert len(Yscale) == verif['lena_Yscale'].shape[0]
    for a, b in zip(Yscale, verif['lena_Yscale']):
        assert np.abs(a-b).max() < TOLERANCE

# vim:sw=4:sts=4:et
