import os
import pytest

from pytest import raises

import numpy as np
from importlib import import_module
from dtcwt.numpy import Transform1d as Transform1d_np
from dtcwt.coeffs import biort, qshift
import tests.datasets as datasets
from .util import skip_if_no_tf
from scipy import stats
from dtcwt.compat import dtwavexfm, dtwaveifm
import dtcwt

PRECISION_DECIMAL = 5
TOLERANCE = 1e-6


@skip_if_no_tf
def setup():
    global mandrill, in_p, pyramid_ops
    global tf, Transform1d, dtwavexfm2, dtwaveifm2, Pyramid_tf
    global np_dtypes, tf_dtypes, stats
    # Import the tensorflow modules
    tf = import_module('tensorflow')
    dtcwt_tf = import_module('dtcwt.tf')
    dtcwt_tf_xfm1 = import_module('dtcwt.tf.transform1d')
    Transform1d = getattr(dtcwt_tf, 'Transform1d')
    Pyramid_tf = getattr(dtcwt_tf, 'Pyramid')
    np_dtypes = getattr(dtcwt_tf_xfm1, 'np_dtypes')
    tf_dtypes = getattr(dtcwt_tf_xfm1, 'tf_dtypes')

    mandrill = datasets.mandrill()
    # Make sure we run tests on cpu rather than gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dtcwt.push_backend('tf')


@skip_if_no_tf
def test_simple():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 3)
    assert len(Yh) == 3


@skip_if_no_tf
def test_simple_with_no_levels():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 0)
    assert len(Yh) == 0


@skip_if_no_tf
def test_simple_with_scale():
    vec = np.random.rand(630)
    Yl, Yh, Yscale = dtwavexfm(vec, 3, include_scale=True)
    assert len(Yh) == 3
    assert len(Yscale) == 3


@skip_if_no_tf
def test_simple_with_scale_and_no_levels():
    vec = np.random.rand(630)
    Yl, Yh, Yscale = dtwavexfm(vec, 0, include_scale=True)
    assert len(Yh) == 0
    assert len(Yscale) == 0


@skip_if_no_tf
def test_perfect_recon():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.max(np.abs(vec_recon - vec)) < TOLERANCE


@skip_if_no_tf
def test_simple_custom_filter():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 4, biort('legall'), qshift('qshift_06'))
    vec_recon = dtwaveifm(Yl, Yh, biort('legall'), qshift('qshift_06'))
    assert np.max(np.abs(vec_recon - vec)) < TOLERANCE


@skip_if_no_tf
def test_single_level():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 1)


@skip_if_no_tf
def test_non_multiple_of_two():
    vec = np.random.rand(631)
    with raises(ValueError):
        Yl, Yh = dtwavexfm(vec, 1)


@skip_if_no_tf
def test_2d():
    Yl, Yh = dtwavexfm(np.random.rand(10,10))


@skip_if_no_tf
def test_integer_input():
    # Check that an integer input is correctly coerced into a floating point
    # array
    Yl, Yh = dtwavexfm([1,2,3,4])
    assert np.any(Yl != 0)


@skip_if_no_tf
def test_integer_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    A = np.array([1,2,3,4], dtype=np.int32)
    Yl, Yh = dtwavexfm(A)
    B = dtwaveifm(Yl, Yh)
    assert np.max(np.abs(A-B)) < 1e-12


@skip_if_no_tf
def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm(np.array([1,2,3,4]).astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))


@skip_if_no_tf
def test_reconstruct():
    # Reconstruction up to tolerance
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < TOLERANCE)


@skip_if_no_tf
def test_reconstruct_2d():
    # Reconstruction up to tolerance
    vec = np.random.rand(630, 20)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < TOLERANCE)


@skip_if_no_tf
def test_float32_input_inv():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm(np.array([1, 2, 3, 4]).astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

    recon = dtwaveifm(Yl, Yh)
    assert np.issubsctype(recon.dtype, np.float32)


@skip_if_no_tf
def test_numpy_in():
    X = np.random.randn(100,100)
    f = Transform1d()
    p = f.forward(X)
    f1 = Transform1d_np()
    p1 = f1.forward(X)
    np.testing.assert_array_almost_equal(
        p.lowpass, p1.lowpass, decimal=PRECISION_DECIMAL)
    for x,y in zip(p.highpasses, p1.highpasses):
        np.testing.assert_array_almost_equal(x,y,decimal=PRECISION_DECIMAL)

    X = np.random.randn(100,100)
    p = f.forward(X, include_scale=True)
    p1 = f1.forward(X, include_scale=True)
    np.testing.assert_array_almost_equal(
        p.lowpass, p1.lowpass, decimal=PRECISION_DECIMAL)
    for x,y in zip(p.highpasses, p1.highpasses):
        np.testing.assert_array_almost_equal(x,y,decimal=PRECISION_DECIMAL)
    for x,y in zip(p.scales, p1.scales):
        np.testing.assert_array_almost_equal(x,y,decimal=PRECISION_DECIMAL)


@skip_if_no_tf
def test_numpy_in_batch():
    X = np.random.randn(5,100,100)

    f = Transform1d()
    p = f.forward_channels(X, include_scale=True)
    f1 = Transform1d_np()
    for i in range(5):
        p1 = f1.forward(X[i], include_scale=True)
        np.testing.assert_array_almost_equal(
            p.lowpass[i], p1.lowpass, decimal=PRECISION_DECIMAL)
        for x,y in zip(p.highpasses, p1.highpasses):
            np.testing.assert_array_almost_equal(
                x[i], y, decimal=PRECISION_DECIMAL)
        for x,y in zip(p.scales, p1.scales):
            np.testing.assert_array_almost_equal(
                x[i], y, decimal=PRECISION_DECIMAL)



# Test end to end with numpy inputs
@skip_if_no_tf
def test_1d_input():
    f = Transform1d()
    X = np.random.randn(100,)
    p = f.forward(X)
    x = f.inverse(p)
    np.testing.assert_array_almost_equal(X,x,decimal=PRECISION_DECIMAL)


@skip_if_no_tf
def test_2d_input():
    f = Transform1d()
    X = np.random.randn(100,100)

    p = f.forward(X)
    x = f.inverse(p)
    np.testing.assert_array_almost_equal(X,x,decimal=PRECISION_DECIMAL)


@skip_if_no_tf
def test_3d_input():
    f = Transform1d()
    X = np.random.randn(5,100,100)

    p = f.forward_channels(X)
    x = f.inverse_channels(p)
    np.testing.assert_array_almost_equal(X,x,decimal=PRECISION_DECIMAL)


# Test end to end with tf inputs
@skip_if_no_tf
def test_2d_input_ph():
    xfm = Transform1d()
    X = np.random.randn(100,)
    X_p = tf.placeholder(tf.float32, [100,])
    p = xfm.forward(X_p)
    x = xfm.inverse(p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_array_almost_equal(
            X, sess.run(x, {X_p:X}), decimal=PRECISION_DECIMAL)

    X = np.random.randn(100,1)
    X_p = tf.placeholder(tf.float32, [None, 100,1])
    p = xfm.forward_channels(X_p)
    x = xfm.inverse_channels(p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_array_almost_equal(
            X, sess.run(x, {X_p:[X]})[0], decimal=PRECISION_DECIMAL)


@skip_if_no_tf
def test_return_type():
    xfm = Transform1d()
    X = np.random.randn(100,100)
    p = xfm.forward(X)
    x = xfm.inverse(p)
    assert x.dtype in np_dtypes
    X = tf.placeholder(tf.float32, [100,100])
    p = xfm.forward(X)
    x = xfm.inverse(p)
    assert x.dtype in tf_dtypes
    X = np.random.randn(5,100,100)
    p = xfm.forward_channels(X)
    x = xfm.inverse_channels(p)
    assert x.dtype in np_dtypes
    X = tf.placeholder(tf.float32, [None, 100,100])
    p = xfm.forward_channels(X)
    x = xfm.inverse_channels(p)
    assert x.dtype in tf_dtypes


@skip_if_no_tf
@pytest.mark.parametrize("test_input,biort,qshift", [
    (datasets.mandrill(),'antonini','qshift_a'),
    (datasets.mandrill()[100:400,40:450],'legall','qshift_a'),
    (datasets.mandrill(),'near_sym_a','qshift_c'),
    (datasets.mandrill()[100:374,30:322],'near_sym_b','qshift_d'),
])
def test_results_match(test_input, biort, qshift):
    """
    Compare forward transform with numpy forward transform for mandrill image
    """
    im = test_input
    f_np = Transform1d_np(biort=biort,qshift=qshift)
    p_np = f_np.forward(im, include_scale=True)

    f_tf = Transform1d(biort=biort,qshift=qshift)
    p_tf = f_tf.forward(im, include_scale=True)

    np.testing.assert_array_almost_equal(
        p_np.lowpass, p_tf.lowpass, decimal=PRECISION_DECIMAL)
    [np.testing.assert_array_almost_equal(
        h_np, h_tf, decimal=PRECISION_DECIMAL) for h_np, h_tf in
        zip(p_np.highpasses, p_tf.highpasses)]
    [np.testing.assert_array_almost_equal(
        s_np, s_tf, decimal=PRECISION_DECIMAL) for s_np, s_tf in
        zip(p_np.scales, p_tf.scales)]


@skip_if_no_tf
@pytest.mark.parametrize("test_input,biort,qshift", [
    (datasets.mandrill(),'antonini','qshift_c'),
    (datasets.mandrill()[99:411,44:460],'near_sym_a','qshift_a'),
    (datasets.mandrill(),'legall','qshift_c'),
    (datasets.mandrill()[100:378,20:322],'near_sym_b','qshift_06'),
])
def test_results_match_inverse(test_input,biort,qshift):
    im = test_input
    f_np = Transform1d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np)

    # Use a zero input and the fwd transform to get the shape of
    # the pyramid easily
    f_tf = Transform1d(biort=biort, qshift=qshift)
    p_tf = f_tf.forward(im, nlevels=4, include_scale=True)

    # Create ops for the inverse transform
    X_tf = f_tf.inverse(p_tf)

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("biort,qshift,gain_mask", [
    ('antonini','qshift_c',stats.bernoulli(0.8).rvs(size=(4))),
    ('near_sym_a','qshift_a',stats.bernoulli(0.8).rvs(size=(4))),
    ('legall','qshift_c',stats.bernoulli(0.8).rvs(size=(4))),
    ('near_sym_b','qshift_06',stats.bernoulli(0.8).rvs(size=(4))),
])
def test_results_match_invmask(biort,qshift,gain_mask):
    im = mandrill

    f_np = Transform1d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np, gain_mask)

    f_tf = Transform1d(biort=biort, qshift=qshift)
    p_tf = f_tf.forward(im, nlevels=4, include_scale=True)
    X_tf = f_tf.inverse(p_tf, gain_mask)

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("test_input, biort, qshift", [
    (datasets.mandrill(), 'antonini', 'qshift_06'),
    (datasets.mandrill()[99:411, 44:460], 'near_sym_b', 'qshift_a'),
    (datasets.mandrill(), 'near_sym_b', 'qshift_c'),
    (datasets.mandrill()[100:378, 20:322], 'near_sym_a', 'qshift_a'),
])
def test_results_match_endtoend(test_input, biort, qshift):
    im = test_input
    f_np = Transform1d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np)

    in_p = tf.placeholder(tf.float32, [im.shape[0], im.shape[1]])
    f_tf = Transform1d(biort=biort, qshift=qshift)
    p_tf = f_tf.forward(in_p, nlevels=4, include_scale=True)
    X = f_tf.inverse(p_tf)
    with tf.Session() as sess:
        X_tf = sess.run(X, feed_dict={in_p: im})

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


# vim:sw=4:sts=4:et
