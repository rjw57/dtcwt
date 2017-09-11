import os
import pytest

from pytest import raises

import numpy as np
from importlib import import_module
import dtcwt
from dtcwt.numpy import Transform2d as Transform2d_np
from dtcwt.coeffs import biort, qshift
from dtcwt.utils import unpack
from scipy import stats
import tests.datasets as datasets
from .util import skip_if_no_tf
import time

PRECISION_DECIMAL = 5


@skip_if_no_tf
def setup():
    # Import some tf only dependencies
    global mandrill, Transform2d, Pyramid
    global tf, np_dtypes, tf_dtypes, dtwavexfm2, dtwaveifm2
    # Import the tensorflow modules
    tf = import_module('tensorflow')
    dtcwt.push_backend('tf')
    Transform2d = getattr(dtcwt, 'Transform2d')
    Pyramid = getattr(dtcwt, 'Pyramid')
    compat = import_module('dtcwt.compat')
    dtwavexfm2 = getattr(compat, 'dtwavexfm2')
    dtwaveifm2 = getattr(compat, 'dtwaveifm2')

    mandrill = datasets.mandrill()
    # Make sure we run tests on cpu rather than gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


@skip_if_no_tf
def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32


@skip_if_no_tf
def test_simple():
    Yl, Yh = dtwavexfm2(mandrill)


@skip_if_no_tf
def test_specific_wavelet():
    Yl, Yh = dtwavexfm2(mandrill, biort=biort('antonini'),
                        qshift=qshift('qshift_06'))


@skip_if_no_tf
def test_1d():
    Yl, Yh = dtwavexfm2(mandrill[0,:])


@skip_if_no_tf
@pytest.mark.skip(reason='Not currently implemented')
def test_3d():
    with raises(ValueError):
        Yl, Yh = dtwavexfm2(np.dstack((mandrill, mandrill)))


@skip_if_no_tf
def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill, include_scale=True)
    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None


@skip_if_no_tf
def test_odd_rows():
    Yl, Yh = dtwavexfm2(mandrill[:509,:])


@skip_if_no_tf
def test_odd_rows_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:], include_scale=True)


@skip_if_no_tf
def test_odd_cols():
    Yl, Yh = dtwavexfm2(mandrill[:,:509])


@skip_if_no_tf
def test_odd_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:509], include_scale=True)


@skip_if_no_tf
def test_odd_rows_and_cols():
    Yl, Yh = dtwavexfm2(mandrill[:,:509])


@skip_if_no_tf
def test_odd_rows_and_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:509], include_scale=True)


@skip_if_no_tf
def test_rot_symm_modified():
    # This test only checks there is no error running these functions,
    # not that they work
    Yl, Yh, Yscale = dtwavexfm2(mandrill, biort='near_sym_b_bp',
                                qshift='qshift_b_bp', include_scale=True)
    dtwaveifm2(Yl, Yh, biort='near_sym_b_bp', qshift='qshift_b_bp')


@skip_if_no_tf
def test_0_levels():
    Yl, Yh = dtwavexfm2(mandrill, nlevels=0)
    np.testing.assert_array_almost_equal(Yl, mandrill, PRECISION_DECIMAL)
    assert len(Yh) == 0


@skip_if_no_tf
def test_0_levels_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill, nlevels=0, include_scale=True)
    np.testing.assert_array_almost_equal(Yl, mandrill, PRECISION_DECIMAL)
    assert len(Yh) == 0
    assert len(Yscale) == 0


@skip_if_no_tf
def test_integer_input():
    # Check that an integer input is correctly coerced into a floating point
    # array
    Yl, Yh = dtwavexfm2([[1,2,3,4], [1,2,3,4]])
    assert np.any(Yl != 0)


@skip_if_no_tf
def test_integer_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    A = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int32)
    Yl, Yh = dtwavexfm2(A)
    B = dtwaveifm2(Yl, Yh)
    assert np.max(np.abs(A - B)) < 1e-5


@skip_if_no_tf
def test_mandrill_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    Yl, Yh = dtwavexfm2(mandrill)
    B = dtwaveifm2(Yl, Yh)
    assert np.max(np.abs(mandrill - B)) < 1e-5


@skip_if_no_tf
def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm2(mandrill.astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))


@skip_if_no_tf
def test_numpy_in():
    X = np.random.randn(100,100)
    f = Transform2d()
    p = f.forward(X)
    f1 = Transform2d_np()
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
@pytest.mark.parametrize("data_format", [
    ("nhw"), ("chw"), ("hwn"), ("hwc")
])
def test_numpy_in_batch(data_format):
    if data_format == "nhw" or data_format == "chw":
        X = np.random.randn(5,100,100)
    else:
        X = np.random.randn(100,100,5)

    f = Transform2d()
    p = f.forward_channels(X, data_format=data_format, include_scale=True)
    f1 = Transform2d_np()
    for i in range(5):
        if data_format == "nhw" or data_format == "chw":
            p1 = f1.forward(X[i], include_scale=True)
            np.testing.assert_array_almost_equal(
                p.lowpass[i], p1.lowpass, decimal=PRECISION_DECIMAL)
            for x,y in zip(p.highpasses, p1.highpasses):
                np.testing.assert_array_almost_equal(
                    x[i], y, decimal=PRECISION_DECIMAL)
            for x,y in zip(p.scales, p1.scales):
                np.testing.assert_array_almost_equal(
                    x[i], y, decimal=PRECISION_DECIMAL)
        else:
            p1 = f1.forward(X[:,:,i], include_scale=True)
            np.testing.assert_array_almost_equal(
                p.lowpass[:,:,i], p1.lowpass, decimal=PRECISION_DECIMAL)
            for x,y in zip(p.highpasses, p1.highpasses):
                np.testing.assert_array_almost_equal(
                    x[:,:,i], y, decimal=PRECISION_DECIMAL)
            for x,y in zip(p.scales, p1.scales):
                np.testing.assert_array_almost_equal(
                    x[:,:,i], y, decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("data_format", [
    ("nhwc"), ("nchw")
])
def test_numpy_batch_ch(data_format):
    if data_format == "nhwc":
        X = np.random.randn(5,100,100,4)
    else:
        X = np.random.randn(5,4,100,100)

    f = Transform2d()
    p = f.forward_channels(X, data_format=data_format, include_scale=True)
    f1 = Transform2d_np()
    for i in range(5):
        for j in range(4):
            if data_format == "nhwc":
                p1 = f1.forward(X[i,:,:,j], include_scale=True)
                np.testing.assert_array_almost_equal(
                    p.lowpass[i,:,:,j], p1.lowpass, decimal=PRECISION_DECIMAL)
                for x,y in zip(p.highpasses, p1.highpasses):
                    np.testing.assert_array_almost_equal(
                        x[i,:,:,j], y, decimal=PRECISION_DECIMAL)
                for x,y in zip(p.scales, p1.scales):
                    np.testing.assert_array_almost_equal(
                        x[i,:,:,j], y, decimal=PRECISION_DECIMAL)
            else:
                p1 = f1.forward(X[i,j], include_scale=True)
                np.testing.assert_array_almost_equal(
                    p.lowpass[i,j], p1.lowpass, decimal=PRECISION_DECIMAL)
                for x,y in zip(p.highpasses, p1.highpasses):
                    np.testing.assert_array_almost_equal(
                        x[i,j], y, decimal=PRECISION_DECIMAL)
                for x,y in zip(p.scales, p1.scales):
                    np.testing.assert_array_almost_equal(
                        x[i,j], y, decimal=PRECISION_DECIMAL)


# Test end to end with numpy inputs
@skip_if_no_tf
def test_2d_input():
    f = Transform2d()
    X = np.random.randn(100,100)
    p = f.forward(X)
    x = f.inverse(p)
    np.testing.assert_array_almost_equal(X,x,decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("data_format", [
    ("nhw"), ("hwn")
])
def test_3d_input(data_format):
    f = Transform2d()
    if data_format == "nhw":
        X = np.random.randn(5,100,100)
    else:
        X = np.random.randn(100,100,5)

    p = f.forward_channels(X,data_format=data_format)
    x = f.inverse_channels(p,data_format=data_format)
    np.testing.assert_array_almost_equal(X,x,decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("data_format", [
    ("nhwc"), ("nchw")
])
def test_4d_input(data_format):
    f = Transform2d()
    if data_format == "nhwc":
        X = np.random.randn(5,100,100,4)
    else:
        X = np.random.randn(5,4,100,100)
    p = f.forward_channels(X,data_format=data_format)
    x = f.inverse_channels(p,data_format=data_format)
    np.testing.assert_array_almost_equal(X,x,decimal=PRECISION_DECIMAL)


# Test end to end with tf inputs
@skip_if_no_tf
def test_2d_input_ph():
    xfm = Transform2d()
    X = np.random.randn(100,100)
    X_p = tf.placeholder(tf.float32, [100,100])
    p = xfm.forward(X_p)
    x = xfm.inverse(p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_array_almost_equal(
            X, sess.run(x, {X_p:X}), decimal=PRECISION_DECIMAL)

    X_p = tf.placeholder(tf.float32, [None, 100,100])
    p = xfm.forward_channels(X_p,data_format="nhw")
    x = xfm.inverse_channels(p,data_format="nhw")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_array_almost_equal(
            X, sess.run(x, {X_p:[X]})[0], decimal=PRECISION_DECIMAL)


# Test end to end with tf inputs
@skip_if_no_tf
def test_3d_input_ph():
    xfm = Transform2d()
    X = np.random.randn(5,100,100)
    X_p = tf.placeholder(tf.float32, [None,100,100])
    p = xfm.forward_channels(X_p,data_format="nhw")
    x = xfm.inverse_channels(p,data_format="nhw")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_array_almost_equal(
            X, sess.run(x, {X_p:X}), decimal=PRECISION_DECIMAL)


@skip_if_no_tf
def test_4d_input_ph():
    xfm = Transform2d()
    X = np.random.randn(5,100,100,4)
    X_p = tf.placeholder(tf.float32, [None,100,100,4])
    p = xfm.forward_channels(X_p,data_format="nhwc")
    x = xfm.inverse_channels(p,data_format="nhwc")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.testing.assert_array_almost_equal(
            X, sess.run(x, {X_p:X}), decimal=PRECISION_DECIMAL)


@skip_if_no_tf
def test_return_type():
    xfm = Transform2d()
    X = np.random.randn(100,100)
    p = xfm.forward(X)
    x = xfm.inverse(p)
    assert x.dtype in np_dtypes
    X = tf.placeholder(tf.float32, [100,100])
    p = xfm.forward(X)
    x = xfm.inverse(p)
    assert x.dtype in tf_dtypes
    xfm = Transform2d()
    X = np.random.randn(5,100,100,4)
    p = xfm.forward_channels(X,data_format="nhwc")
    x = xfm.inverse_channels(p,data_format="nhwc")
    assert x.dtype in np_dtypes
    xfm = Transform2d()
    X = tf.placeholder(tf.float32, [None, 100,100,4])
    p = xfm.forward_channels(X,data_format="nhwc")
    x = xfm.inverse_channels(p,data_format="nhwc")
    assert x.dtype in tf_dtypes


@skip_if_no_tf
@pytest.mark.parametrize("test_input,biort,qshift", [
    (datasets.mandrill(),'antonini','qshift_a'),
    (datasets.mandrill()[100:400,40:450],'legall','qshift_a'),
    (datasets.mandrill(),'near_sym_a','qshift_c'),
    (datasets.mandrill()[100:375,30:322],'near_sym_b','qshift_d'),
    (datasets.mandrill(),'near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match(test_input, biort, qshift):
    """
    Compare forward transform with numpy forward transform for mandrill image
    """
    im = test_input
    f_np = Transform2d_np(biort=biort,qshift=qshift)
    p_np = f_np.forward(im, include_scale=True)

    f_tf = Transform2d(biort=biort,qshift=qshift)
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
    (datasets.mandrill()[100:411,44:460],'near_sym_a','qshift_a'),
    (datasets.mandrill(),'legall','qshift_c'),
    (datasets.mandrill()[100:378,20:322],'near_sym_b','qshift_06'),
    (datasets.mandrill(),'near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match_inverse(test_input,biort,qshift):
    im = test_input
    f_np = Transform2d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np)

    # Use a zero input and the fwd transform to get the shape of
    # the pyramid easily
    f_tf = Transform2d(biort=biort, qshift=qshift)
    p_tf = f_tf.forward(im, nlevels=4, include_scale=True)

    # Create ops for the inverse transform
    X_tf = f_tf.inverse(p_tf)

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("biort,qshift,gain_mask", [
    ('antonini','qshift_c',stats.bernoulli(0.8).rvs(size=(6,4))),
    ('near_sym_a','qshift_a',stats.bernoulli(0.8).rvs(size=(6,4))),
    ('legall','qshift_c',stats.bernoulli(0.8).rvs(size=(6,4))),
    ('near_sym_b','qshift_06',stats.bernoulli(0.8).rvs(size=(6,4))),
    ('near_sym_b_bp', 'qshift_b_bp',stats.bernoulli(0.8).rvs(size=(6,4)))
])
def test_results_match_invmask(biort,qshift,gain_mask):
    im = mandrill

    f_np = Transform2d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np, gain_mask)

    f_tf = Transform2d(biort=biort, qshift=qshift)
    p_tf = f_tf.forward(im, nlevels=4, include_scale=True)
    X_tf = f_tf.inverse(p_tf, gain_mask)

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("test_input, biort, qshift", [
    (datasets.mandrill(), 'antonini', 'qshift_06'),
    (datasets.mandrill()[100:411, 44:460], 'near_sym_b', 'qshift_a'),
    (datasets.mandrill(), 'near_sym_b', 'qshift_c'),
    (datasets.mandrill()[100:378, 20:322], 'near_sym_a', 'qshift_a'),
    (datasets.mandrill(), 'near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match_endtoend(test_input, biort, qshift):
    im = test_input
    f_np = Transform2d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np)

    in_p = tf.placeholder(tf.float32, [im.shape[0], im.shape[1]])
    f_tf = Transform2d(biort=biort, qshift=qshift)
    p_tf = f_tf.forward(in_p, nlevels=4, include_scale=True)
    X = f_tf.inverse(p_tf)
    with tf.Session() as sess:
        X_tf = sess.run(X, feed_dict={in_p: im})

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


@skip_if_no_tf
@pytest.mark.parametrize("data_format", [
    ("nhwc"),
    ("nchw"),
])
def test_forward_channels(data_format):
    batch = 5
    c = 3
    nlevels = 3
    sess = tf.Session()

    if data_format == "nhwc":
        ims = np.random.randn(batch, 100, 100, c)
        in_p = tf.placeholder(tf.float32, [None, 100, 100, c])
    else:
        ims = np.random.randn(batch, c, 100, 100)
        in_p = tf.placeholder(tf.float32, [None, c, 100, 100])

    # Transform a set of images with forward_channels
    f_tf = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    start = time.time()
    Yl, Yh, Yscale = unpack(
        f_tf.forward_channels(in_p, nlevels=nlevels, data_format=data_format,
                              include_scale=True), 'tf')

    Yl, Yh, Yscale = sess.run([Yl, Yh, Yscale], {in_p: ims})
    print("That took {:.2f}s".format(time.time() - start))

    # Now do it channel by channel
    in_p2 = tf.placeholder(tf.float32, [None, 100, 100])
    p_tf = f_tf.forward_channels(in_p2, data_format="nhw", nlevels=nlevels,
                                 include_scale=True)
    for i in range(c):
        if data_format == "nhwc":
            Yl1, Yh1, Yscale1 = sess.run([p_tf.lowpass_op,
                                          p_tf.highpasses_ops,
                                          p_tf.scales_ops],
                                         {in_p2: ims[:,:,:,i]})
            np.testing.assert_array_almost_equal(
                Yl[:,:,:,i], Yl1, decimal=PRECISION_DECIMAL)
            for j in range(nlevels):
                np.testing.assert_array_almost_equal(
                    Yh[j][:,:,:,i,:], Yh1[j], decimal=PRECISION_DECIMAL)
                np.testing.assert_array_almost_equal(
                    Yscale[j][:,:,:,i], Yscale1[j], decimal=PRECISION_DECIMAL)
        else:
            Yl1, Yh1, Yscale1 = sess.run([p_tf.lowpass_op,
                                          p_tf.highpasses_ops,
                                          p_tf.scales_ops],
                                         {in_p2: ims[:,i]})
            np.testing.assert_array_almost_equal(
                Yl[:,i], Yl1, decimal=PRECISION_DECIMAL)
            for j in range(nlevels):
                np.testing.assert_array_almost_equal(
                    Yh[j][:,i], Yh1[j], decimal=PRECISION_DECIMAL)
                np.testing.assert_array_almost_equal(
                    Yscale[j][:,i], Yscale1[j], decimal=PRECISION_DECIMAL)
    sess.close()


@skip_if_no_tf
@pytest.mark.parametrize("data_format", [
    ("nhwc"),
    ("nchw"),
])
def test_inverse_channels(data_format):
    batch = 5
    c = 3
    nlevels = 3
    sess = tf.Session()

    # Create the tensors of the right shape by calling the forward function
    if data_format == "nhwc":
        ims = np.random.randn(batch, 100, 100, c)
        in_p = tf.placeholder(tf.float32, [None, 100, 100, c])
        f_tf = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
        Yl, Yh = unpack(
            f_tf.forward_channels(in_p, nlevels=nlevels,
                                  data_format=data_format), 'tf')
    else:
        ims = np.random.randn(batch, c, 100, 100)
        in_p = tf.placeholder(tf.float32, [None, c, 100, 100])
        f_tf = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
        Yl, Yh = unpack(f_tf.forward_channels(
            in_p, nlevels=nlevels, data_format=data_format), 'tf')

    # Call the inverse_channels function
    start = time.time()
    X = f_tf.inverse_channels(Pyramid(Yl, Yh), data_format=data_format)
    X, Yl, Yh = sess.run([X, Yl, Yh], {in_p: ims})
    print("That took {:.2f}s".format(time.time() - start))

    # Now do it channel by channel
    in_p2 = tf.zeros((batch, 100, 100), tf.float32)
    p_tf = f_tf.forward_channels(in_p2, nlevels=nlevels, data_format="nhw",
                                 include_scale=False)
    X_t = f_tf.inverse_channels(p_tf,data_format="nhw")
    for i in range(c):
        Yh1 = []
        if data_format == "nhwc":
            Yl1 = Yl[:,:,:,i]
            for j in range(nlevels):
                Yh1.append(Yh[j][:,:,:,i])
        else:
            Yl1 = Yl[:,i]
            for j in range(nlevels):
                Yh1.append(Yh[j][:,i])

        # Use the eval_inv function to feed the data into the right variables
        sess.run(tf.global_variables_initializer())
        X1 = sess.run(X_t, {p_tf.lowpass_op: Yl1, p_tf.highpasses_ops: Yh1})

        if data_format == "nhwc":
            np.testing.assert_array_almost_equal(
                X[:,:,:,i], X1, decimal=PRECISION_DECIMAL)
        else:
            np.testing.assert_array_almost_equal(
                X[:,i], X1, decimal=PRECISION_DECIMAL)

    sess.close()

# vim:sw=4:sts=4:et
