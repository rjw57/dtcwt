import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm, dtwaveifm

@attr('transform')
def test_reconstruct():
    # Reconstruction up to tolerance
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)
    vec_recon = dtwaveifm(Yl, Yh)
    assert np.all(np.abs(vec_recon - vec) < 1e-7)

# vim:sw=4:sts=4:et
