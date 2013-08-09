import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm, dtwaveifm, biort, qshift

@attr('transform')
def test_simple():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 3)
    assert len(Yh) == 3

@attr('transform')
def test_simple_with_no_levels():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 0)
    assert len(Yh) == 0

@attr('transform')
def test_simple_with_scale():
    vec = np.random.rand(630)
    Yl, Yh, Yscale = dtwavexfm(vec, 3, include_scale=True)
    assert len(Yh) == 3
    assert len(Yscale) == 3

@attr('transform')
def test_simple_with_scale_and_no_levels():
    vec = np.random.rand(630)
    Yl, Yh, Yscale = dtwavexfm(vec, 0, include_scale=True)
    assert len(Yh) == 0
    assert len(Yscale) == 0

@attr('transform')
def test_simple_custom_filter():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 4, biort('legall'), qshift('qshift_06'))
    vec_recon = dtwaveifm(Yl, Yh, biort('legall'), qshift('qshift_06'))
    assert np.max(np.abs(vec_recon - vec)) < 1e-8

@attr('transform')
def test_single_level():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 1)

@raises(ValueError)
def test_non_multiple_of_two():
    vec = np.random.rand(631)
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

# vim:sw=4:sts=4:et
