import os
from pytest import raises

import numpy as np
from dtcwt.compat import dtwavexfm, dtwaveifm

TOLERANCE = 1e-12

def test_reconstruct():
    # Reconstruction up to tolerance
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < TOLERANCE)

def test_reconstruct_2d():
    # Reconstruction up to tolerance
    vec = np.random.rand(630, 20)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < TOLERANCE)

def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm(np.array([1, 2, 3, 4]).astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

    recon = dtwaveifm(Yl, Yh)
    assert np.issubsctype(recon.dtype, np.float32)

# vim:sw=4:sts=4:et
