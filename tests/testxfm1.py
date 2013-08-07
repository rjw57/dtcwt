import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt import dtwavexfm

@attr('transform')
def test_simple():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec)

@attr('transform')
def test_single_level():
    vec = np.random.rand(630)
    Yl, Yh = dtwavexfm(vec, 1)

@raises(ValueError)
def test_non_multiple_of_two():
    vec = np.random.rand(631)
    Yl, Yh = dtwavexfm(vec, 1)

@raises(ValueError)
def test_2d():
    Yl, Yh = dtwavexfm(np.random.rand(10,10))

# vim:sw=4:sts=4:et
