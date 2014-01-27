import os

import numpy as np

import dtcwt
from dtcwt.registration import *

def setup():
    global f1, f2
    frames = np.load(os.path.join(os.path.dirname(__file__), 'traffic.npz'))
    f1 = frames['f1']
    f2 = frames['f2']

def test_frames_loaded():
    assert f1.shape == (576, 768)
    assert f1.min() >= 0
    assert f1.max() <= 1
    assert f1.dtype == np.float64

    assert f2.shape == (576, 768)
    assert f2.min() >= 0
    assert f2.max() <= 1
    assert f2.dtype == np.float64

def test_estimate_flor():
    nlevels = 6
    Yl1, Yh1 = dtcwt.dtwavexfm2(f1, nlevels=nlevels)
    Yl2, Yh2 = dtcwt.dtwavexfm2(f2, nlevels=nlevels)
    avecs = estimateflow(Yh1, Yh2)

    # Make sure warped frame 1 has lower mean overlap error than non-warped
    warped_f1 = warp(f1, avecs, method='bilinear')
    assert np.mean(np.abs(warped_f1 - f2)) < np.mean(np.abs(f1-f2))

