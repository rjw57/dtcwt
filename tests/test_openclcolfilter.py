import os

import numpy as np
from dtcwt.coeffs import biort, qshift
from dtcwt.opencl.lowlevel import colfilter
from dtcwt.numpy.lowlevel import colfilter as colfilter_gold

from .util import assert_almost_equal, skip_if_no_cl
import tests.datasets as datasets

def setup():
    global mandrill
    mandrill = datasets.mandrill()

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

@skip_if_no_cl
def test_odd_size():
    y = colfilter(mandrill, (-1,2,-1))
    assert y.shape == mandrill.shape

    z = colfilter_gold(mandrill, (-1,2,-1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_even_size():
    y = colfilter(mandrill, (-1,1))
    assert y.shape == (mandrill.shape[0]+1, mandrill.shape[1])

    z = colfilter_gold(mandrill, (-1,1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_odd_size():
    y = colfilter(mandrill, (-1,2,-1))
    assert y.shape == mandrill.shape

    z = colfilter_gold(mandrill, (-1,2,-1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_qshift():
    y = colfilter(mandrill, qshift('qshift_a')[0])
    assert y.shape == (mandrill.shape[0]+1, mandrill.shape[1])

    z = colfilter_gold(mandrill, qshift('qshift_a')[0])
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_biort():
    y = colfilter(mandrill, biort('antonini')[0])
    assert y.shape == mandrill.shape

    z = colfilter_gold(mandrill, biort('antonini')[0])
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_even_size():
    y = colfilter(np.zeros_like(mandrill), (-1,1))
    assert y.shape == (mandrill.shape[0]+1, mandrill.shape[1])
    assert not np.any(y[:] != 0.0)

    z = colfilter_gold(np.zeros_like(mandrill), (-1,1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_odd_size_non_array():
    y = colfilter(mandrill.tolist(), (-1,2,-1))
    assert y.shape == mandrill.shape

    z = colfilter_gold(mandrill.tolist(), (-1,2,-1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_even_size_non_array():
    y = colfilter(mandrill.tolist(), (-1,1))
    assert y.shape == (mandrill.shape[0]+1, mandrill.shape[1])

    z = colfilter_gold(mandrill.tolist(), (-1,1))
    assert_almost_equal(y, z)

# vim:sw=4:sts=4:et
