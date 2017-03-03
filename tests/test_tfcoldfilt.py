import os

import pytest
from dtcwt.tf.lowlevel import _HAVE_TF as HAVE_TF
pytest.mark.skipif(not HAVE_TF, reason="Tensorflow not present")

import numpy as np
import tensorflow as tf
from dtcwt.coeffs import biort, qshift
from dtcwt.tf.lowlevel import coldfilt
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt

from pytest import raises

import tests.datasets as datasets

def setup():
    global mandrill, mandrill_t
    mandrill = datasets.mandrill()
    mandrill_t = tf.expand_dims(tf.constant(mandrill, dtype=tf.float32),axis=0)

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32
    assert mandrill_t.get_shape() == (1, 512, 512)

def test_odd_filter():
    with raises(ValueError):
        coldfilt(mandrill_t, (-1,2,-1), (-1,2,1))

def test_different_size():
    with raises(ValueError):
        coldfilt(mandrill_t, (-0.5,-1,2,1,0.5), (-1,2,-1))

def test_bad_input_size():
    with raises(ValueError):
        coldfilt(mandrill_t[:,:511,:], (-1,1), (1,-1))

def test_good_input_size():
    coldfilt(mandrill_t[:,:,:511], (-1,1), (1,-1))

def test_good_input_size_non_orthogonal():
    coldfilt(mandrill_t[:,:,:511], (1,1), (1,1))

def test_output_size():
    y_op = coldfilt(mandrill_t, (-1,1), (1,-1))
    assert y_op.shape[1:] == (mandrill.shape[0]/2, mandrill.shape[1])

def test_equal_numpy_qshift():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    ref = np_coldfilt(mandrill, ha, hb)
    y_op = coldfilt(mandrill_t, ha, hb)
    with tf.Session() as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

# vim:sw=4:sts=4:et
