import os

import numpy as np
from dtcwt import biort, qshift
from dtcwt.backend.backend_opencl.lowlevel import colfilter
from dtcwt.lowlevel import colfilter as colfilter_gold

from .util import assert_almost_equal, skip_if_no_cl
import tests.datasets as datasets

def setup():
    global lena
    lena = datasets.lena()

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

@skip_if_no_cl
def test_odd_size():
    y = colfilter(lena, (-1,2,-1))
    assert y.shape == lena.shape

    z = colfilter_gold(lena, (-1,2,-1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_even_size():
    y = colfilter(lena, (-1,1))
    assert y.shape == (lena.shape[0]+1, lena.shape[1])

    z = colfilter_gold(lena, (-1,1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_odd_size():
    y = colfilter(lena, (-1,2,-1))
    assert y.shape == lena.shape

    z = colfilter_gold(lena, (-1,2,-1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_qshift():
    y = colfilter(lena, qshift('qshift_a')[0])
    assert y.shape == (lena.shape[0]+1, lena.shape[1])

    z = colfilter_gold(lena, qshift('qshift_a')[0])
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_biort():
    y = colfilter(lena, biort('antonini')[0])
    assert y.shape == lena.shape

    z = colfilter_gold(lena, biort('antonini')[0])
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_even_size():
    y = colfilter(np.zeros_like(lena), (-1,1))
    assert y.shape == (lena.shape[0]+1, lena.shape[1])
    assert not np.any(y[:] != 0.0)

    z = colfilter_gold(np.zeros_like(lena), (-1,1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_odd_size_non_array():
    y = colfilter(lena.tolist(), (-1,2,-1))
    assert y.shape == lena.shape

    z = colfilter_gold(lena.tolist(), (-1,2,-1))
    assert_almost_equal(y, z)

@skip_if_no_cl
def test_even_size_non_array():
    y = colfilter(lena.tolist(), (-1,1))
    assert y.shape == (lena.shape[0]+1, lena.shape[1])

    z = colfilter_gold(lena.tolist(), (-1,1))
    assert_almost_equal(y, z)
	
# vim:sw=4:sts=4:et
