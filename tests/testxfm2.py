import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm2, dtwaveifm2, biort, qshift

TOLERANCE = 1e-12

def setup():
    global lena
    lena = np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

def test_lena_loaded():
    assert lena.shape == (512, 512)
    assert lena.min() >= 0
    assert lena.max() <= 1
    assert lena.dtype == np.float32

@attr('transform')
def test_simple():
    Yl, Yh = dtwavexfm2(lena)

@attr('transform')
def test_specific_wavelet():
    Yl, Yh = dtwavexfm2(lena, biort=biort('antonini'), qshift=qshift('qshift_06'))

def test_1d():
    Yl, Yh = dtwavexfm2(lena[0,:])

@raises(ValueError)
def test_3d():
    Yl, Yh = dtwavexfm2(np.dstack((lena, lena)))

def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena, include_scale=True)

    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None

def test_odd_rows():
    Yl, Yh = dtwavexfm2(lena[:509,:])

def test_odd_rows_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena[:509,:], include_scale=True)

def test_odd_cols():
    Yl, Yh = dtwavexfm2(lena[:,:509])

def test_odd_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena[:509,:509], include_scale=True)

def test_odd_rows_and_cols():
    Yl, Yh = dtwavexfm2(lena[:,:509])

def test_odd_rows_and_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena[:509,:509], include_scale=True)

def test_0_levels():
    Yl, Yh = dtwavexfm2(lena, nlevels=0)
    assert np.all(np.abs(Yl - lena) < TOLERANCE)
    assert len(Yh) == 0

def test_0_levels_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(lena, nlevels=0, include_scale=True)
    assert np.all(np.abs(Yl - lena) < TOLERANCE)
    assert len(Yh) == 0
    assert len(Yscale) == 0

def test_integer_input():
    # Check that an integer input is correctly coerced into a floating point
    # array
    Yl, Yh = dtwavexfm2([[1,2,3,4], [1,2,3,4]])
    assert np.any(Yl != 0)

def test_integer_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    A = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int32)
    Yl, Yh = dtwavexfm2(A)
    B = dtwaveifm2(Yl, Yh)
    assert np.max(np.abs(A-B)) < 1e-5

def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm2(lena.astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

# vim:sw=4:sts=4:et
