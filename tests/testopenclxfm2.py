import os
from pytest import raises

import numpy as np
from dtcwt.coeffs import biort, qshift
from dtcwt.compat import dtwavexfm2 as dtwavexfm2_np, dtwaveifm2
from dtcwt.opencl.transform2d import dtwavexfm2 as dtwavexfm2_cl

from .util import assert_almost_equal, skip_if_no_cl
import tests.datasets as datasets

TOLERANCE = 1e-12
GOLD_TOLERANCE = 1e-5

def setup():
    global mandrill
    mandrill = datasets.mandrill()

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

def _compare_transforms(A, B):
    Yl_A, Yh_A = A
    Yl_B, Yh_B = B
    assert_almost_equal(Yl_A, Yl_B, tolerance=GOLD_TOLERANCE)
    for x, y in zip(Yh_A, Yh_B):
        assert_almost_equal(x, y, tolerance=GOLD_TOLERANCE)

@skip_if_no_cl
def test_simple():
    _compare_transforms(dtwavexfm2_np(mandrill), dtwavexfm2_cl(mandrill))

@skip_if_no_cl
def test_specific_wavelet():
    a = dtwavexfm2_np(mandrill, biort=biort('antonini'), qshift=qshift('qshift_06'))
    b = dtwavexfm2_cl(mandrill, biort=biort('antonini'), qshift=qshift('qshift_06'))
    _compare_transforms(a, b)

@skip_if_no_cl
def test_1d():
    a = dtwavexfm2_np(mandrill[0,:])
    b = dtwavexfm2_cl(mandrill[0,:])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_3d():
    with raises(ValueError):
        Yl, Yh = dtwavexfm2_cl(np.dstack((mandrill, mandrill)))

@skip_if_no_cl
def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2_cl(mandrill, include_scale=True)

    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None

@skip_if_no_cl
@skip_if_no_cl
def test_odd_rows():
    a = dtwavexfm2_np(mandrill[:509,:])
    b = dtwavexfm2_cl(mandrill[:509,:])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_odd_cols():
    a = dtwavexfm2_np(mandrill[:,:509])
    b = dtwavexfm2_cl(mandrill[:,:509])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_odd_rows_and_cols():
    a = dtwavexfm2_np(mandrill[:509,:509])
    b = dtwavexfm2_cl(mandrill[:509,:509])
    _compare_transforms(a, b)

@skip_if_no_cl
def test_0_levels():
    a = dtwavexfm2_np(mandrill, nlevels=0)
    b = dtwavexfm2_cl(mandrill, nlevels=0)
    _compare_transforms(a, b)

@skip_if_no_cl
def test_modified():
    a = dtwavexfm2_np(mandrill, biort=biort('near_sym_b_bp'), qshift=qshift('qshift_b_bp'))
    b = dtwavexfm2_cl(mandrill, biort=biort('near_sym_b_bp'), qshift=qshift('qshift_b_bp'))
    _compare_transforms(a, b)

# vim:sw=4:sts=4:et
