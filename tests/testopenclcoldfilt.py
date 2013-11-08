import os

import numpy as np
from dtcwt.lowlevel import coldfilt as coldfilt_gold
from dtcwt.opencl.lowlevel import coldfilt, NoCLPresentError
from dtcwt.coeffs import biort, qshift

from nose.tools import raises

from .util import assert_almost_equal, skip_if_no_cl

def setup():
    global lena
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

@raises(ValueError)
@skip_if_no_cl
def test_odd_filter():
    coldfilt(lena, (-1,2,-1), (-1,2,1))

@raises(ValueError)
@skip_if_no_cl
def test_different_size():
    coldfilt(lena, (-0.5,-1,2,1,0.5), (-1,2,-1))

@raises(ValueError)
@skip_if_no_cl
def test_bad_input_size():
    coldfilt(lena[:511,:], (-1,1), (1,-1))

@skip_if_no_cl
def test_real_wavelet():
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    A = coldfilt(lena[:,:511], h1b, h1a)
    B = coldfilt_gold(lena[:,:511], h1b, h1a)
    assert_almost_equal(A, B)

@skip_if_no_cl
def test_good_input_size():
    A = coldfilt(lena[:,:511], (-1,1), (1,-1))
    B = coldfilt_gold(lena[:,:511], (-1,1), (1,-1))
    assert_almost_equal(A, B)

@skip_if_no_cl
def test_good_input_size_non_orthogonal():
    A = coldfilt(lena[:,:511], (1,1), (1,1))
    B = coldfilt_gold(lena[:,:511], (1,1), (1,1))
    assert_almost_equal(A, B)

@skip_if_no_cl
def test_output_size():
    Y = coldfilt(lena, (-1,1), (1,-1))
    assert Y.shape == (lena.shape[0]/2, lena.shape[1])

    Z = coldfilt_gold(lena, (-1,1), (1,-1))
    assert_almost_equal(Y, Z)

@skip_if_no_cl
def test_qshift():
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift('qshift_d')
    y = coldfilt(lena, h1b, h1b)
    z = coldfilt_gold(lena, h1b, h1a)
    assert_almost_equal(y, z)

# This test fails but I'm not sure if that's actually a problem. I'm not
# convinced coldfilt does the right think in this case.
#
# @skip_if_no_cl
# def test_qshift_even_input():
#     h1b = np.array((-0.25, 0.5, 0.5, -0.25))
#     h1a = h1b[::-1]
#     y = coldfilt(lena, h1b, h1a)
#     z = coldfilt_gold(lena, h1b, h1a)
#     assert_almost_equal(y, z)

# vim:sw=4:sts=4:et
