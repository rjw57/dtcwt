import os

import pytest

import numpy as np
from dtcwt.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt
from importlib import import_module

from pytest import raises
from .util import skip_if_no_tf

import tests.datasets as datasets

@skip_if_no_tf
def test_setup():
    global mandrill, mandrill_t, tf, coldfilt
    tf = import_module('tensorflow')
    lowlevel = import_module('dtcwt.tf.lowlevel')
    coldfilt = getattr(lowlevel, 'coldfilt')

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
def test_odd_filter():
    with raises(ValueError):
        coldfilt(mandrill_t, (-1,2,-1), (-1,2,1))

@skip_if_no_tf
def test_different_size():
    with raises(ValueError):
        coldfilt(mandrill_t, (-0.5,-1,2,1,0.5), (-1,2,-1))

@skip_if_no_tf
def test_bad_input_size():
    with raises(ValueError):
        coldfilt(mandrill_t[:,:511,:], (-1,1), (1,-1))

@skip_if_no_tf
def test_good_input_size():
    coldfilt(mandrill_t[:,:,:511], (-1,1), (1,-1))

@skip_if_no_tf
def test_good_input_size_non_orthogonal():
    coldfilt(mandrill_t[:,:,:511], (1,1), (1,1))

@skip_if_no_tf
def test_output_size():
    y_op = coldfilt(mandrill_t, (-1,1), (1,-1))
    assert y_op.shape[1:] == (mandrill.shape[0]/2, mandrill.shape[1])

@skip_if_no_tf
@pytest.mark.skip(reason='Cant pad by more than half the dimension of the input')
def test_equal_small_in():
    ha = qshift('qshift_b')[0]
    hb = qshift('qshift_b')[1]
    im = mandrill[0:4,0:4]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = np_coldfilt(im, ha, hb)
    y_op = coldfilt(im_t, ha, hb)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

@skip_if_no_tf
def test_equal_numpy_qshift1():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    ref = np_coldfilt(mandrill, ha, hb)
    y_op = coldfilt(mandrill_t, ha, hb)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

@skip_if_no_tf
def test_equal_numpy_qshift2():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im = mandrill[:508, :502]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = np_coldfilt(im, ha, hb)
    y_op = coldfilt(im_t, ha, hb)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

# vim:sw=4:sts=4:et
