import os
from nose.tools import raises
from nose.plugins.attrib import attr

import numpy as np
from dtcwt.compat import dtwavexfm2, dtwaveifm2
from dtcwt.coeffs import biort, qshift
import tests.datasets as datasets

TOLERANCE = 1e-12

def setup():
    global mandrill, mandrill_crop
    mandrill = datasets.mandrill().astype(np.float64)
    mandrill_crop = mandrill[:233, :301]

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float64

@attr('transform')
def test_reconstruct():
    # Reconstruction up to tolerance
    Yl, Yh = dtwavexfm2(mandrill)
    mandrill_recon = dtwaveifm2(Yl, Yh)
    assert np.all(np.abs(mandrill_recon - mandrill) < TOLERANCE)

@attr('transform')
def test_reconstruct_crop():
    # Reconstruction up to tolerance
    Yl_crop, Yh_crop = dtwavexfm2(mandrill_crop)
    mandrill_recon = dtwaveifm2(Yl_crop, Yh_crop)[:mandrill_crop.shape[0], :mandrill_crop.shape[1]]
    assert np.all(np.abs(mandrill_recon - mandrill_crop) < TOLERANCE)

@attr('transform')
def test_reconstruct_custom_filter():
    # Reconstruction up to tolerance
    Yl, Yh = dtwavexfm2(mandrill, 4, biort('legall'), qshift('qshift_06'))
    mandrill_recon = dtwaveifm2(Yl, Yh, biort('legall'), qshift('qshift_06'))
    assert np.all(np.abs(mandrill_recon - mandrill) < TOLERANCE)

def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm2(mandrill.astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

    mandrill_recon = dtwaveifm2(Yl, Yh)
    assert np.issubsctype(mandrill_recon.dtype, np.float32)


# vim:sw=4:sts=4:et
