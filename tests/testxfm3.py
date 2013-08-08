import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm3, dtwaveifm3

GRID_SIZE=32
SPHERE_RAD=25

def setup():
    global sphere

    grid = np.arange(-(GRID_SIZE>>1), (GRID_SIZE>>1))
    X, Y, Z = np.meshgrid(grid, grid, grid)

    Y *= 1.2
    Z /= 1.2

    r = np.sqrt(X*X + Y*Y + Z*Z)
    sphere = np.where(r <= SPHERE_RAD, 1.0, 0.0)

def test_sphere():
    # Check general aspects of sphere are OK
    assert sphere.shape == (GRID_SIZE,GRID_SIZE,GRID_SIZE)
    assert sphere.min() == 0
    assert sphere.max() == 1

def test_simple_level_1_xfm():
    # Just tests that the transform broadly works and gives expected size output
    Yl, Yh = dtwavexfm3(sphere, 1)
    assert Yl.shape == (GRID_SIZE,GRID_SIZE,GRID_SIZE)
    assert len(Yh) == 1

def test_simple_level_1_recon():
    # Test for perfect reconstruction with 1 level
    Yl, Yh = dtwavexfm3(sphere, 1)
    sphere_recon = dtwaveifm3(Yl, Yh)
    assert sphere.size == sphere_recon.size
    assert np.max(np.abs(sphere - sphere_recon)) < 1e-11

# vim:sw=4:sts=4:et
