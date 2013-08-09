import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm, dtwaveifm

TOLERANCE = 1e-12

@attr('transform')
def test_reconstruct():
    # Reconstruction up to tolerance
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < TOLERANCE)

@attr('transform')
def test_reconstruct_2d():
    # Reconstruction up to tolerance
    vec = np.random.rand(630, 20)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < TOLERANCE)

# vim:sw=4:sts=4:et
