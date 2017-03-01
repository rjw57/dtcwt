import os

import numpy as np
import tensorflow as tf
from dtcwt.coeffs import biort, qshift
from dtcwt.tf.lowlevel import colfilter

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

def test_odd_size():
    h = tf.constant([-1,2,-1], dtype=tf.float32)
    y_op = colfilter(mandrill_t, h)
    assert y_op.get_shape()[1:] == mandrill.shape

def test_even_size():
    h = tf.constant([-1,-1], dtype=tf.float32)
    y_op = colfilter(mandrill_t, h)
    assert y_op.get_shape()[1:] == (mandrill.shape[0]+1, mandrill.shape[1])

def test_qshift():
    h = tf.constant(qshift('qshift_a')[0], dtype=tf.float32)
    y_op = colfilter(mandrill, h)
    assert y_op.get_shape()[1:] == (mandrill.shape[0]+1, mandrill.shape[1])

def test_biort():
    h = tf.constant(biort('antonini')[0], dtype=tf.float32)
    y_op = colfilter(mandrill, h)
    assert y_op.get_shape()[1:] == mandrill.shape

def test_even_size():
    h = tf.constant([-1,-1], dtype=tf.float32)
    y_op = colfilter(mandrill_t, h)
    assert y_op.get_shape()[1:] == (mandrill.shape[0]+1, mandrill.shape[1])
    with tf.Session() as sess:
        y = sess.run(y_op)
    assert not np.any(y[:] != 0.0)


# vim:sw=4:sts=4:et
