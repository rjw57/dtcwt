import os

import numpy as np
from dtcwt.numpy.lowlevel import colifilt

from nose.tools import raises

import tests.datasets as datasets

def setup():
    global mandrill
    mandrill = datasets.mandrill()

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

@raises(ValueError)
def test_odd_filter():
    colifilt(mandrill, (-1,2,-1), (-1,2,1))

@raises(ValueError)
def test_different_size_h():
    colifilt(mandrill, (-1,2,1), (-0.5,-1,2,-1,0.5))

def test_zero_input():
    Y = colifilt(np.zeros_like(mandrill), (-1,1), (1,-1))
    assert np.all(Y[:0] == 0)

@raises(ValueError)
def test_bad_input_size():
    colifilt(mandrill[:511,:], (-1,1), (1,-1))

def test_good_input_size():
    colifilt(mandrill[:,:511], (-1,1), (1,-1))

def test_output_size():
    Y = colifilt(mandrill, (-1,1), (1,-1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])

def test_non_orthogonal_input():
    Y = colifilt(mandrill, (1,1), (1,1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])

def test_output_size_non_mult_4():
    Y = colifilt(mandrill, (-1,0,0,1), (1,0,0,-1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])

def test_non_orthogonal_input_non_mult_4():
    Y = colifilt(mandrill, (1,0,0,1), (1,0,0,1))
    assert Y.shape == (mandrill.shape[0]*2, mandrill.shape[1])

# vim:sw=4:sts=4:et
