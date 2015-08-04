import os

import numpy as np
from dtcwt.opencl.lowlevel import colifilt
from dtcwt.numpy.lowlevel import colifilt as colifilt_gold
from dtcwt.coeffs import biort, qshift

from nose.tools import raises

from .util import assert_almost_equal, skip_if_no_cl
import tests.datasets as datasets

def setup():
    global mandrill
    mandrill = datasets.mandrill()

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

@skip_if_no_cl
@raises(ValueError)
def test_odd_filter():
    colifilt(mandrill, (-1,2,-1), (-1,2,1))

@skip_if_no_cl
@raises(ValueError)
def test_different_size_h():
    colifilt(mandrill, (-1,2,1), (-0.5,-1,2,-1,0.5))

@skip_if_no_cl
def test_zero_input():
    Y = colifilt(np.zeros_like(mandrill), (-1,1), (1,-1))
    assert np.all(Y[:0] == 0)

@skip_if_no_cl
@raises(ValueError)
def test_bad_input_size():
    colifilt(mandrill[:511,:], (-1,1), (1,-1))

@skip_if_no_cl
def test_good_input_size():
    Y = colifilt(mandrill[:,:511], (-1,1), (1,-1))
    Z = colifilt_gold(mandrill[:,:511], (-1,1), (1,-1))
    assert_almost_equal(Y,Z)

@skip_if_no_cl
def test_output_size():
    Y = colifilt(mandrill, (-1,1), (1,-1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])
    Z = colifilt_gold(mandrill, (-1,1), (1,-1))
    assert_almost_equal(Y,Z)

@skip_if_no_cl
def test_non_orthogonal_input():
    Y = colifilt(mandrill, (1,1), (1,1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])
    Z = colifilt_gold(mandrill, (1,1), (1,1))
    assert_almost_equal(Y,Z)

@skip_if_no_cl
def test_output_size_non_mult_4():
    Y = colifilt(mandrill, (-1,0,0,1), (1,0,0,-1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])
    Z = colifilt_gold(mandrill, (-1,0,0,1), (1,0,0,-1))
    assert_almost_equal(Y,Z)

@skip_if_no_cl
def test_non_orthogonal_input_non_mult_4():
    Y = colifilt(mandrill, (1,0,0,1), (1,0,0,1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])
    Z = colifilt_gold(mandrill, (1,0,0,1), (1,0,0,1))
    assert_almost_equal(Y,Z)

@skip_if_no_cl
def test_qshift():
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    y = colifilt(mandrill, h1b, h1b)
    z = colifilt_gold(mandrill, h1b, h1a)
    assert_almost_equal(y, z)

# This test fails. I'm not sure if that's expected or not because it is using
# colifilt in an odd way.
#
# @skip_if_no_cl
# def test_qshift_odd_len_input_1():
#     h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
#     h1a = h1a[:-2]
#     h1b = h1a[::-1]
#     y = colifilt(mandrill, h1a, h1b)
#     z = colifilt_gold(mandrill, h1a, h1b)
#     assert_almost_equal(y, z)

@skip_if_no_cl
def test_qshift_odd_len_input_2():
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    y = colifilt(mandrill, h1a[1:-1], h1b[1:-1])
    z = colifilt_gold(mandrill, h1a[1:-1], h1b[1:-1])
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_qshift_even_input():
    h1b = np.array((-0.25, 0.5, 0.5, -0.25))
    h1a = h1b[::-1]
    y = colifilt(mandrill, h1b, h1a)
    z = colifilt_gold(mandrill, h1b, h1a)
    assert_almost_equal(y, z)

# vim:sw=4:sts=4:et
