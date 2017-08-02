import numpy as np
from dtcwt.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter
from importlib import import_module

from tests.util import skip_if_no_tf
import tests.datasets as datasets


@skip_if_no_tf
def test_setup():
    global mandrill, mandrill_t, tf, colfilter
    tf = import_module('tensorflow')
    lowlevel = import_module('dtcwt.tf.lowlevel')
    colfilter = getattr(lowlevel, 'colfilter')

    mandrill = datasets.mandrill()
    mandrill_t = tf.expand_dims(tf.constant(mandrill, dtype=tf.float32),axis=0)


@skip_if_no_tf
def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32
    assert mandrill_t.get_shape() == (1, 512, 512)


@skip_if_no_tf
def test_odd_size():
    y_op = colfilter(mandrill_t, [-1,2,-1])
    assert y_op.get_shape()[1:] == mandrill.shape


@skip_if_no_tf
def test_even_size():
    y_op = colfilter(mandrill_t, [-1,-1])
    assert y_op.get_shape()[1:] == (mandrill.shape[0]+1, mandrill.shape[1])


@skip_if_no_tf
def test_qshift():
    h = qshift('qshift_a')[0]
    y_op = colfilter(mandrill_t, h)
    assert y_op.get_shape()[1:] == (mandrill.shape[0]+1, mandrill.shape[1])


@skip_if_no_tf
def test_biort():
    h = biort('antonini')[0]
    y_op = colfilter(mandrill_t, h)
    assert y_op.get_shape()[1:] == mandrill.shape


@skip_if_no_tf
def test_even_size_batch():
    zero_t = tf.zeros([1, mandrill.shape[0], mandrill.shape[1]], tf.float32)
    y_op = colfilter(zero_t, [-1,1])
    assert y_op.get_shape()[1:] == (mandrill.shape[0]+1, mandrill.shape[1])
    with tf.Session() as sess:
        y = sess.run(y_op)
    assert not np.any(y[:] != 0.0)


@skip_if_no_tf
def test_equal_small_in():
    h = qshift('qshift_b')[0]
    im = mandrill[0:4,0:4]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = np_colfilter(im, h)
    y_op = colfilter(im_t, h)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


@skip_if_no_tf
def test_equal_numpy_biort1():
    h = biort('near_sym_b')[0]
    ref = np_colfilter(mandrill, h)
    y_op = colfilter(mandrill_t, h)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


@skip_if_no_tf
def test_equal_numpy_biort2():
    h = biort('near_sym_b')[0]
    im = mandrill[52:407, 30:401]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = np_colfilter(im, h)
    y_op = colfilter(im_t, h)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


@skip_if_no_tf
def test_equal_numpy_qshift1():
    h = qshift('qshift_c')[0]
    ref = np_colfilter(mandrill, h)
    y_op = colfilter(mandrill_t, h)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


@skip_if_no_tf
def test_equal_numpy_qshift2():
    h = qshift('qshift_c')[0]
    im = mandrill[52:407, 30:401]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = np_colfilter(im, h)
    y_op = colfilter(im_t, h)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

# vim:sw=4:sts=4:et
