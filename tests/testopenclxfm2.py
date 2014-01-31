import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import biort, qshift
from dtcwt import dtwavexfm2 as dtwavexfm2_np, dtwaveifm2
from dtcwt.backend.backend_opencl.transform2d import dtwavexfm2 as dtwavexfm2_cl

from .util import assert_almost_equal, skip_if_no_cl
import tests.datasets as datasets

TOLERANCE = 1e-12
GOLD_TOLERANCE = 1e-5

def setup():
    global lena
    lena = datasets.lena()

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

def _compare_transforms(A, B):
    Yl_A, Yh_A = A
    Yl_B, Yh_B = B
    assert_almost_equal(Yl_A, Yl_B, tolerance=GOLD_TOLERANCE)
    for x, y in zip(Yh_A, Yh_B):
        assert_almost_equal(x, y, tolerance=GOLD_TOLERANCE)

@skip_if_no_cl
@attr('transform')
def test_simple():
    _compare_transforms(dtwavexfm2_np(lena), dtwavexfm2_cl(lena))

@skip_if_no_cl
@attr('transform')
def test_specific_wavelet():
    a = dtwavexfm2_np(lena, biort=biort('antonini'), qshift=qshift('qshift_06'))
    b = dtwavexfm2_cl(lena, biort=biort('antonini'), qshift=qshift('qshift_06'))
    _compare_transforms(a, b)

@skip_if_no_cl
def test_1d():
    a = dtwavexfm2_np(lena[0,:])
    b = dtwavexfm2_cl(lena[0,:])
    _compare_transforms(a, b)

@skip_if_no_cl
@raises(ValueError)
def test_3d():
    Yl, Yh = dtwavexfm2_cl(np.dstack((lena, lena)))

@skip_if_no_cl
def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2_cl(lena, include_scale=True)

    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None

@skip_if_no_cl
@skip_if_no_cl
def test_odd_rows():
    a = dtwavexfm2_np(lena[:509,:])
    b = dtwavexfm2_cl(lena[:509,:])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_odd_cols():
    a = dtwavexfm2_np(lena[:,:509])
    b = dtwavexfm2_cl(lena[:,:509])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_odd_rows_and_cols():
    a = dtwavexfm2_np(lena[:509,:509])
    b = dtwavexfm2_cl(lena[:509,:509])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_0_levels():
    a = dtwavexfm2_np(lena, nlevels=0)
    b = dtwavexfm2_cl(lena, nlevels=0)
    _compare_transforms(a, b)

@skip_if_no_cl
@attr('transform')
def test_modified():
    a = dtwavexfm2_np(lena, biort=biort('near_sym_b_bp'), qshift=qshift('qshift_b_bp'))
    b = dtwavexfm2_cl(lena, biort=biort('near_sym_b_bp'), qshift=qshift('qshift_b_bp'))
    _compare_transforms(a, b)

# vim:sw=4:sts=4:et
