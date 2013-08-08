import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm3, dtwaveifm3, biort, qshift

GRID_SIZE=32
SPHERE_RAD=0.4 * GRID_SIZE
TOLERANCE = 1e-12

def setup():
    global ellipsoid

    grid = np.arange(-(GRID_SIZE>>1), (GRID_SIZE>>1))
    X, Y, Z = np.meshgrid(grid, grid, grid)

    Y *= 1.2
    Z *= 1.4

    r = np.sqrt(X*X + Y*Y + Z*Z)
    ellipsoid = np.where(r <= SPHERE_RAD, 1.0, 0.0).astype(np.float64)

def test_ellipsoid():
    # Check general aspects of ellipsoid are OK
    assert ellipsoid.shape == (GRID_SIZE,GRID_SIZE,GRID_SIZE)
    assert ellipsoid.min() == 0
    assert ellipsoid.max() == 1

def test_simple_level_1_xfm():
    # Just tests that the transform broadly works and gives expected size output
    Yl, Yh = dtwavexfm3(ellipsoid, 1)
    assert Yl.shape == (GRID_SIZE,GRID_SIZE,GRID_SIZE)
    assert len(Yh) == 1

def test_simple_level_1_recon():
    # Test for perfect reconstruction with 1 level
    Yl, Yh = dtwavexfm3(ellipsoid, 1)
    ellipsoid_recon = dtwaveifm3(Yl, Yh)
    assert ellipsoid.size == ellipsoid_recon.size
    assert np.max(np.abs(ellipsoid - ellipsoid_recon)) < TOLERANCE

def test_simple_level_1_recon_haar():
    # Test for perfect reconstruction with 1 level and Haar wavelets

    # Form Haar wavelets
    h0 = np.array((1.0, 1.0))
    g0 = h0
    h0 = h0 / np.sum(h0)
    g0 = g0 / np.sum(g0)
    h1 = g0 * np.cumprod(-np.ones_like(g0))
    g1 = -h0 * np.cumprod(-np.ones_like(h0))

    haar = (h0, g0, h1, g1)

    Yl, Yh = dtwavexfm3(ellipsoid, 1, biort=haar)
    ellipsoid_recon = dtwaveifm3(Yl, Yh, biort=haar)
    assert ellipsoid.size == ellipsoid_recon.size
    print(np.max(np.abs(ellipsoid - ellipsoid_recon)))
    assert np.max(np.abs(ellipsoid - ellipsoid_recon)) < TOLERANCE

def test_simple_level_2_xfm():
    # Just tests that the transform broadly works and gives expected size output
    Yl, Yh = dtwavexfm3(ellipsoid, 2)
    assert Yl.shape == (GRID_SIZE>>1,GRID_SIZE>>1,GRID_SIZE>>1)
    assert len(Yh) == 2

def test_simple_level_2_recon():
    # Test for perfect reconstruction with 2 levels
    Yl, Yh = dtwavexfm3(ellipsoid, 2)
    ellipsoid_recon = dtwaveifm3(Yl, Yh)
    assert ellipsoid.size == ellipsoid_recon.size
    assert np.max(np.abs(ellipsoid - ellipsoid_recon)) < TOLERANCE

def test_simple_level_4_xfm():
    # Just tests that the transform broadly works and gives expected size output
    Yl, Yh = dtwavexfm3(ellipsoid, 4)
    assert Yl.shape == (GRID_SIZE>>3,GRID_SIZE>>3,GRID_SIZE>>3)
    assert len(Yh) == 4

def test_simple_level_4_recon():
    # Test for perfect reconstruction with 3 levels
    Yl, Yh = dtwavexfm3(ellipsoid, 4)
    ellipsoid_recon = dtwaveifm3(Yl, Yh)
    assert ellipsoid.size == ellipsoid_recon.size
    assert np.max(np.abs(ellipsoid - ellipsoid_recon)) < TOLERANCE

def test_simple_level_4_recon_custom_wavelets():
    # Test for perfect reconstruction with 3 levels
    b = biort('legall')
    q = qshift('qshift_06')
    Yl, Yh = dtwavexfm3(ellipsoid, 4, biort=b, qshift=q)
    ellipsoid_recon = dtwaveifm3(Yl, Yh, biort=b, qshift=q)
    assert ellipsoid.size == ellipsoid_recon.size
    assert np.max(np.abs(ellipsoid - ellipsoid_recon)) < TOLERANCE

def test_simple_level_4_xfm_ext_mode_8():
    # Just tests that the transform broadly works and gives expected size output
    crop_ellipsoid = ellipsoid[:62,:58,:54]
    Yl, Yh = dtwavexfm3(crop_ellipsoid, 4, ext_mode=8)
    assert len(Yh) == 4

def test_simple_level_4_recon_ext_mode_8():
    # Test for perfect reconstruction with 3 levels
    crop_ellipsoid = ellipsoid[:62,:58,:54]
    Yl, Yh = dtwavexfm3(crop_ellipsoid, 4, ext_mode=8)
    ellipsoid_recon = dtwaveifm3(Yl, Yh)
    assert crop_ellipsoid.size == ellipsoid_recon.size
    assert np.max(np.abs(crop_ellipsoid - ellipsoid_recon)) < TOLERANCE

def test_simple_level_4_xfm_ext_mode_4():
    # Just tests that the transform broadly works and gives expected size output
    crop_ellipsoid = ellipsoid[:62,:54,:58]
    Yl, Yh = dtwavexfm3(crop_ellipsoid, 4, ext_mode=4)
    assert len(Yh) == 4

def test_simple_level_4_recon_ext_mode_4():
    # Test for perfect reconstruction with 3 levels
    crop_ellipsoid = ellipsoid[:62,:54,:58]
    Yl, Yh = dtwavexfm3(crop_ellipsoid, 4, ext_mode=4)
    ellipsoid_recon = dtwaveifm3(Yl, Yh)
    assert crop_ellipsoid.size == ellipsoid_recon.size
    assert np.max(np.abs(crop_ellipsoid - ellipsoid_recon)) < TOLERANCE

# vim:sw=4:sts=4:et
