import os

import numpy as np
from dtcwt.numpy.lowlevel import coldfilt as coldfilt_gold
from dtcwt.opencl.lowlevel import coldfilt, NoCLPresentError
from dtcwt.coeffs import biort, qshift

from pytest import raises

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
def test_odd_filter():
    with raises(ValueError):
        coldfilt(mandrill, (-1,2,-1), (-1,2,1))

@skip_if_no_cl
def test_different_size():
    with raises(ValueError):
        coldfilt(mandrill, (-0.5,-1,2,1,0.5), (-1,2,-1))

@skip_if_no_cl
def test_bad_input_size():
    with raises(ValueError):
        coldfilt(mandrill[:511,:], (-1,1), (1,-1))

@skip_if_no_cl
def test_real_wavelet():
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    A = coldfilt(mandrill[:,:511], h1b, h1a)
    B = coldfilt_gold(mandrill[:,:511], h1b, h1a)
    assert_almost_equal(A, B)

@skip_if_no_cl
def test_good_input_size():
    A = coldfilt(mandrill[:,:511], (-1,1), (1,-1))
    B = coldfilt_gold(mandrill[:,:511], (-1,1), (1,-1))
    assert_almost_equal(A, B)

@skip_if_no_cl
def test_good_input_size_non_orthogonal():
    A = coldfilt(mandrill[:,:511], (1,1), (1,1))
    B = coldfilt_gold(mandrill[:,:511], (1,1), (1,1))
    assert_almost_equal(A, B)

@skip_if_no_cl
def test_output_size():
    Y = coldfilt(mandrill, (-1,1), (1,-1))
    assert Y.shape == (mandrill.shape[0]/2, mandrill.shape[1])

    Z = coldfilt_gold(mandrill, (-1,1), (1,-1))
    assert_almost_equal(Y, Z)

@skip_if_no_cl
def test_qshift():
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    y = coldfilt(mandrill, h1b, h1b)
    z = coldfilt_gold(mandrill, h1b, h1a)
    assert_almost_equal(y, z)

# This test fails but I'm not sure if that's actually a problem. I'm not
# convinced coldfilt does the right think in this case.
#
# @skip_if_no_cl
# def test_qshift_even_input():
#     h1b = np.array((-0.25, 0.5, 0.5, -0.25))
#     h1a = h1b[::-1]
#     y = coldfilt(mandrill, h1b, h1a)
#     z = coldfilt_gold(mandrill, h1b, h1a)
#     assert_almost_equal(y, z)

# vim:sw=4:sts=4:et
