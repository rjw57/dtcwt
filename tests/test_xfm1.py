import os
from pytest import raises

import numpy as np
from dtcwt.compat import dtwavexfm, dtwaveifm
from dtcwt.coeffs import biort, qshift

TOLERANCE = 1e-12

def test_simple():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 3)
    assert len(Yh) == 3

def test_simple_with_no_levels():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 0)
    assert len(Yh) == 0

def test_simple_with_scale():
    vec = np.random.rand(630)
    Yl, Yh, Yscale = dtwavexfm(vec, 3, include_scale=True)
    assert len(Yh) == 3
    assert len(Yscale) == 3

def test_simple_with_scale_and_no_levels():
    vec = np.random.rand(630)
    Yl, Yh, Yscale = dtwavexfm(vec, 0, include_scale=True)
    assert len(Yh) == 0
    assert len(Yscale) == 0

def test_perfect_recon():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.max(np.abs(vec_recon - vec)) < TOLERANCE

def test_simple_custom_filter():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 4, biort('legall'), qshift('qshift_06'))
    vec_recon = dtwaveifm(Yl, Yh, biort('legall'), qshift('qshift_06'))
    assert np.max(np.abs(vec_recon - vec)) < TOLERANCE

def test_single_level():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 1)

def test_non_multiple_of_two():
    vec = np.random.rand(631)
    with raises(ValueError):
        Yl, Yh = dtwavexfm(vec, 1)

def test_2d():
    Yl, Yh = dtwavexfm(np.random.rand(10,10))

def test_integer_input():
    # Check that an integer input is correctly coerced into a floating point
    # array
    Yl, Yh = dtwavexfm([1,2,3,4])
    assert np.any(Yl != 0)

def test_integer_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    A = np.array([1,2,3,4], dtype=np.int32)
    Yl, Yh = dtwavexfm(A)
    B = dtwaveifm(Yl, Yh)
    assert np.max(np.abs(A-B)) < 1e-12

def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm(np.array([1,2,3,4]).astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

# vim:sw=4:sts=4:et
