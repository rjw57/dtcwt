import os

import numpy as np
from dtcwt.lowlevel import colifilt

from nose.tools import raises

def setup():
    global lena
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

@raises(ValueError)
def test_odd_filter():
    colifilt(lena, (-1,2,-1), (-1,2,1))

@raises(ValueError)
def test_different_size_h():
    colifilt(lena, (-1,2,1), (-0.5,-1,2,-1,0.5))

def test_zero_input():
    Y = colifilt(np.zeros_like(lena), (-1,1), (1,-1))
    assert np.all(Y[:0] == 0)

@raises(ValueError)
def test_bad_input_size():
    colifilt(lena[:511,:], (-1,1), (1,-1))

def test_good_input_size():
    colifilt(lena[:,:511], (-1,1), (1,-1))

def test_output_size():
    Y = colifilt(lena, (-1,1), (1,-1))
    assert Y.shape == (lena.shape[0]*2, lena.shape[1])

def test_non_orthogonal_input():
    Y = colifilt(lena, (1,1), (1,1))
    assert Y.shape == (lena.shape[0]*2, lena.shape[1])

def test_output_size_non_mult_4():
    Y = colifilt(lena, (-1,0,0,1), (1,0,0,-1))
    assert Y.shape == (lena.shape[0]*2, lena.shape[1])

def test_non_orthogonal_input_non_mult_4():
    Y = colifilt(lena, (1,0,0,1), (1,0,0,1))
    assert Y.shape == (lena.shape[0]*2, lena.shape[1])

# vim:sw=4:sts=4:et
