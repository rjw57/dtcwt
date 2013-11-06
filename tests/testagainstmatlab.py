import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm2, dtwaveifm2, biort, qshift
from dtcwt.lowlevel import coldfilt, colifilt

from .util import assert_almost_equal

# We allow a little more tolerance for comparison with MATLAB
TOLERANCE = 1e-5

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
    assert_almost_equal(A, verif['lena_coldfilt'])

def test_coldfilt():
    h0o, g0o, h1o, g1o = biort('near_sym_b')
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    A = colifilt(lena, g0b, g0a)
    assert_almost_equal(A, verif['lena_colifilt'])

def test_dtwavexfm2():
    Yl, Yh, Yscale = dtwavexfm2(lena, 4, 'near_sym_a', 'qshift_a', include_scale=True)
    assert_almost_equal(Yl, verif['lena_Yl'], tolerance=TOLERANCE)

    for idx, a in enumerate(Yh):
        assert_almost_equal(a, verif['lena_Yh_{0}'.format(idx)], tolerance=TOLERANCE)

    for idx, a in enumerate(Yscale):
        assert_almost_equal(a, verif['lena_Yscale_{0}'.format(idx)], tolerance=TOLERANCE)

# vim:sw=4:sts=4:et
