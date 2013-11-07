import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm2, dtwaveifm2, biort, qshift
from dtcwt.lowlevel import coldfilt, colifilt

from .util import assert_almost_equal, summarise_mat

## IMPORTANT NOTE ##

# These tests match only a 'summary' matrix from MATLAB which is formed by
# dividing a matrix into 9 parts thusly:
#
#  A | B | C
# ---+---+---
#  D | E | F
# ---+---+---
#  G | H | I
#
# Where A, C, G and I are NxN and N is some agreed 'apron' size. E is replaced
# my it's element-wise mean and thus becomes 1x1. The remaining matrices are
# replaced by the element-wise mean along the apropriate axis to result in a
# (2N+1) x (2N+1) matrix. These matrices are compared.
#
# The rationale for this summary is that the corner matrices preserve
# interesting edge-effects and some actual values whereas the interior matrices
# preserve at least some information on their contents. Storing such a summary
# matrix greatly reduces the amount of storage required.

# Summary matching requires greater tolerance
TOLERANCE = 1e-5

def assert_almost_equal_to_summary(a, summary, *args, **kwargs):
    assert_almost_equal(summarise_mat(a), summary, *args, **kwargs)

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
    assert_almost_equal_to_summary(A, verif['lena_coldfilt'], tolerance=TOLERANCE)

def test_coldfilt():
    h0o, g0o, h1o, g1o = biort('near_sym_b')
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    A = colifilt(lena, g0b, g0a)
    assert_almost_equal_to_summary(A, verif['lena_colifilt'], tolerance=TOLERANCE)

def test_dtwavexfm2():
    Yl, Yh, Yscale = dtwavexfm2(lena, 4, 'near_sym_a', 'qshift_a', include_scale=True)
    assert_almost_equal_to_summary(Yl, verif['lena_Yl'], tolerance=TOLERANCE)

    for idx, a in enumerate(Yh):
        assert_almost_equal_to_summary(a, verif['lena_Yh_{0}'.format(idx)], tolerance=TOLERANCE)

    for idx, a in enumerate(Yscale):
        assert_almost_equal_to_summary(a, verif['lena_Yscale_{0}'.format(idx)], tolerance=TOLERANCE)

# vim:sw=4:sts=4:et
