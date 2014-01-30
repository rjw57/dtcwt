import os

import numpy as np

import dtcwt
from dtcwt.backend.backend_numpy import Transform2d
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

def test_estimatereg():
    nlevels = 6
    trans = Transform2d()
    t1 = trans.forward(f1, nlevels=nlevels)
    t2 = trans.forward(f2, nlevels=nlevels)
    avecs = estimatereg(t1, t2)

    # Make sure warped frame 1 has lower mean overlap error than non-warped
    warped_f1 = warp(f1, avecs, method='bilinear')
    assert np.mean(np.abs(warped_f1 - f2)) < np.mean(np.abs(f1-f2))

