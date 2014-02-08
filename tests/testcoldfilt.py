import os

import numpy as np
from dtcwt.numpy.lowlevel import coldfilt

from nose.tools import raises

import tests.datasets as datasets

def setup():
    global lena
    lena = datasets.lena()

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

@raises(ValueError)
def test_odd_filter():
    coldfilt(lena, (-1,2,-1), (-1,2,1))

@raises(ValueError)
def test_different_size():
    coldfilt(lena, (-0.5,-1,2,1,0.5), (-1,2,-1))

@raises(ValueError)
def test_bad_input_size():
    coldfilt(lena[:511,:], (-1,1), (1,-1))

def test_good_input_size():
    coldfilt(lena[:,:511], (-1,1), (1,-1))

def test_good_input_size_non_orthogonal():
    coldfilt(lena[:,:511], (1,1), (1,1))

def test_output_size():
    Y = coldfilt(lena, (-1,1), (1,-1))
    assert Y.shape == (lena.shape[0]/2, lena.shape[1])

# vim:sw=4:sts=4:et
