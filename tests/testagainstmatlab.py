import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np

from scipy.io import loadmat
from dtcwt import dtwavexfm2, dtwavexfm3, dtwaveifm2, dtwavexfm2b, dtwaveifm2b, biort, qshift
from dtcwt.lowlevel import coldfilt, colifilt
from dtcwt.sampling import rescale_highpass

from dtcwt.backend.base import TransformDomainSignal, ReconstructedSignal

from .util import assert_almost_equal, summarise_mat, summarise_cube, assert_percentile_almost_equal

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
# by it's element-wise mean and thus becomes 1x1. The remaining matrices are
# replaced by the element-wise mean along the apropriate axis to result in a
# (2N+1) x (2N+1) matrix. These matrices are compared.
#
# The rationale for this summary is that the corner matrices preserve
# interesting edge-effects and some actual values whereas the interior matrices
# preserve at least some information on their contents. Storing such a summary
# matrix greatly reduces the amount of storage required.

# Summary matching requires greater tolerance

# We allow a little more tolerance for comparison with MATLAB
TOLERANCE = 1e-5

def assert_almost_equal_to_summary(a, summary, *args, **kwargs):
    assert_almost_equal(summarise_mat(a), summary, *args, **kwargs)

def assert_percentile_almost_equal_to_summary(a, summary, *args, **kwargs):
    assert_percentile_almost_equal(summarise_mat(a), summary, *args, **kwargs)

def assert_almost_equal_to_summary_cube(a, summary, *args, **kwargs):
    assert_almost_equal(summarise_cube(a), summary, *args, **kwargs)

def assert_percentile_almost_equal_to_summary_cube(a, summary, *args, **kwargs):
    assert_percentile_almost_equal(summarise_cube(a), summary, *args, **kwargs)

def setup():
    global lena
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

    global qbgn
    qbgn = loadmat(os.path.join(os.path.dirname(__file__), 'qbgn.mat'))['qbgn']

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

def test_dtwavexfm2b():
    Yl, Yh, Yscale = dtwavexfm2b(lena, 4, 'near_sym_b_bp', 'qshift_b_bp', include_scale=True)
    assert_almost_equal_to_summary(Yl, verif['lena_Ylb'], tolerance=TOLERANCE)

    for idx, a in enumerate(Yh):
        assert_almost_equal_to_summary(a, verif['lena_Yhb_{0}'.format(idx)], tolerance=TOLERANCE)

    for idx, a in enumerate(Yscale):
        assert_almost_equal_to_summary(a, verif['lena_Yscaleb_{0}'.format(idx)], tolerance=TOLERANCE)

def test_rescale_highpass():
    # N.B we can only test bilinear rescaling since cpxinterb2b doesn't support Lanczos
    Yl, Yh = dtwavexfm2b(lena, 3, 'near_sym_a', 'qshift_a')
    X = Yh[2]
    Xrescale = rescale_highpass(X, (X.shape[0]*3, X.shape[1]*3), 'bilinear')

    # Almost equal in the rescale case is a hard thing to quantify. Due to the
    # slight differences in interpolation method the odd pixel can very by
    # quite an amount. Use a percentile approach to look at the bigger picture.
    assert_percentile_almost_equal_to_summary(Xrescale, verif['lena_upsample'], 60, tolerance=TOLERANCE)

def test_transform3d_numpy():
    from dtcwt.backend.backend_numpy import Transform3d
    transform = Transform3d(biort='near_sym_a',qshift='qshift_b')
    td_signal = transform.forward(qbgn, nlevels=3, include_scale=True, discard_level_1=False)
    Yl, Yh, Yscale = td_signal.lowpass, td_signal.subbands, td_signal.scales
    #assert_almost_equal_to_summary_cube(Yl, verif['qbgn_Yl'], tolerance=TOLERANCE)
    for idx, a in enumerate(Yh):
        assert_almost_equal_to_summary_cube(a, verif['qbgn_Yh_{0}'.format(idx)], tolerance=TOLERANCE)

    for idx, a in enumerate(Yscale):
        assert_almost_equal_to_summary_cube(a, verif['qbgn_Yscale_{0}'.format(idx)], tolerance=TOLERANCE)

# vim:sw=4:sts=4:et
