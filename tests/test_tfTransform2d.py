import os
import pytest
from dtcwt.tf.lowlevel import _HAVE_TF as HAVE_TF
pytest.mark.skipif(not HAVE_TF, reason="Tensorflow not present")

from pytest import raises

import numpy as np
import tensorflow as tf
from dtcwt.tf import Transform2d, dtwavexfm2
from dtcwt.numpy import Transform2d as Transform2d_np
from dtcwt.numpy import Pyramid
from dtcwt.coeffs import biort, qshift
import tests.datasets as datasets
#from .util import skip_if_no_tf

PRECISION_DECIMAL = 5

def setup():
    global mandrill, in_p, pyramid_ops
    mandrill = datasets.mandrill()
    in_p = tf.placeholder(tf.float32, [None, 512, 512])
    f = Transform2d()
    pyramid_ops = f.forward(in_p, include_scale=True)

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

def test_simple():
    Yl, Yh = dtwavexfm2(mandrill)

def test_specific_wavelet():
    Yl, Yh = dtwavexfm2(mandrill, biort=biort('antonini'), qshift=qshift('qshift_06'))

@pytest.mark.skip(reason='Cant pad by more than half the dimension of the input')
def test_1d():
    Yl, Yh = dtwavexfm2(mandrill[0,:])

@pytest.mark.skip(reason='Not currently implemented')
def test_3d():
    with raises(ValueError):
        Yl, Yh = dtwavexfm2(np.dstack((mandrill, mandrill)))

def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill, include_scale=True)

    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None

def test_odd_rows():
    Yl, Yh = dtwavexfm2(mandrill[:509,:])

def test_odd_rows_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:], include_scale=True)

def test_odd_cols():
    Yl, Yh = dtwavexfm2(mandrill[:,:509])

def test_odd_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:509], include_scale=True)

def test_odd_rows_and_cols():
    Yl, Yh = dtwavexfm2(mandrill[:,:509])

def test_odd_rows_and_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:509], include_scale=True)

@pytest.mark.skip(reason='Inverse not currently implemented')
def test_rot_symm_modified():
    # This test only checks there is no error running these functions, not that they work
    Yl, Yh, Yscale = dtwavexfm2(mandrill, biort='near_sym_b_bp', qshift='qshift_b_bp', include_scale=True)
    Z = dtwaveifm2(Yl, Yh, biort='near_sym_b_bp', qshift='qshift_b_bp')

def test_0_levels():
    Yl, Yh = dtwavexfm2(mandrill, nlevels=0)
    np.testing.assert_array_almost_equal(Yl, mandrill, PRECISION_DECIMAL)
    assert len(Yh) == 0

def test_0_levels_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill, nlevels=0, include_scale=True)
    np.testing.assert_array_almost_equal(Yl, mandrill, PRECISION_DECIMAL)
    assert len(Yh) == 0
    assert len(Yscale) == 0

@pytest.mark.skip(reason='Cant pad by more than half the dimension of the input')
def test_integer_input():
    # Check that an integer input is correctly coerced into a floating point
    # array
    Yl, Yh = dtwavexfm2([[1,2,3,4], [1,2,3,4]])
    assert np.any(Yl != 0)

@pytest.mark.skip(reason='Inverse not currently implemented')
def test_integer_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    A = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int32)
    Yl, Yh = dtwavexfm2(A)
    B = dtwaveifm2(Yl, Yh)
    assert np.max(np.abs(A-B)) < 1e-5

def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm2(mandrill.astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

def test_multiple_inputs():
    y = pyramid_ops.eval(mandrill)
    y3 = pyramid_ops.eval([mandrill, mandrill, mandrill])
    assert y3.lowpass.shape == (3, *y.lowpass.shape)
    for hi3, hi in zip(y3.highpasses, y.highpasses):
        assert hi3.shape == (3, *hi.shape)
    for s3, s in zip(y3.scales, y.scales):
        assert s3.shape == (3, *s.shape)

def test_results_match1():
    f_np = Transform2d_np()
    p_np = f_np.forward(mandrill, include_scale=True) 
    
    p_tf = pyramid_ops.eval(mandrill)
    
    np.testing.assert_array_almost_equal(
            p_np.lowpass, p_tf.lowpass, decimal=PRECISION_DECIMAL) 
    [np.testing.assert_array_almost_equal(
            h_np, h_tf, decimal=PRECISION_DECIMAL) for h_np, h_tf in 
            zip(p_np.highpasses, p_tf.highpasses)]
    [np.testing.assert_array_almost_equal(
            s_np, s_tf, decimal=PRECISION_DECIMAL) for s_np, s_tf in 
            zip(p_np.scales, p_tf.scales)]

def test_results_match2():
    im = mandrill[100:400,50:450]
    f_np = Transform2d_np(biort='near_sym_b', qshift='qshift_c')
    p_np = f_np.forward(im, nlevels=3, include_scale=True) 

    f_tf = Transform2d(biort='near_sym_b', qshift='qshift_c')
    p_tf = f_tf.forward(im, nlevels=3, include_scale=True)
    
    np.testing.assert_array_almost_equal(
            p_np.lowpass, p_tf.lowpass, decimal=PRECISION_DECIMAL) 
    [np.testing.assert_array_almost_equal(
            h_np, h_tf, decimal=PRECISION_DECIMAL) for h_np, h_tf in 
            zip(p_np.highpasses, p_tf.highpasses)]
    [np.testing.assert_array_almost_equal(
            s_np, s_tf, decimal=PRECISION_DECIMAL) for s_np, s_tf in 
            zip(p_np.scales, p_tf.scales)]
   
# vim:sw=4:sts=4:et
