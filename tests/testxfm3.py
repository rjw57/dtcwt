import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm3, dtwaveifm3

def setup():
    global sphere
    X, Y, Z = np.meshgrid(np.arange(-63,64), np.arange(-63,64), np.arange(-63,64))

    r = np.sqrt(X*X + Y*Y + Z*Z)
    sphere = np.where(r <= 55, 1.0, 0.0)

def test_sphere():
    # Check general aspects of sphere are OK
    assert sphere.shape == (127,127,127)
    assert sphere.min() >= 0
    assert sphere.max() <= 1

    # Check volume of sphere is ok to within 5%
    sphere_vol = (4.0/3.0) * np.pi * 55*55*55
    assert np.abs(np.sum(sphere.flatten()) - sphere_vol) < 5e-2*sphere_vol

# vim:sw=4:sts=4:et
