#!/usr/bin/env python
"""
An example of image registration via the DTCWT.

This script demonstrates some methods for image registration using the DTCWT.

"""

from __future__ import division, print_function

import itertools
import logging
import os

from matplotlib.pyplot import *
import numpy as np

import dtcwt
import dtcwt.sampling
from dtcwt.registration import *

logging.basicConfig(level=logging.INFO)

def register_frames(filename):
    # Load test images
    logging.info('Loading frames from "{0}"'.format(filename))
    test_frames = np.load(filename)
    f1 = test_frames['f1']
    f2 = test_frames['f2']

    # Take the DTCWT of both frames.
    logging.info('Taking DTCWT')
    nlevels = 6
    Yl1, Yh1 = dtcwt.dtwavexfm2(f1, nlevels=nlevels)
    Yl2, Yh2 = dtcwt.dtwavexfm2(f2, nlevels=nlevels)

    # Solve for transform
    logging.info('Finding flow')
    avecs = estimateflow(Yh1, Yh2)

    logging.info('Computing warped image')
    warped_f1 = warp(f1, avecs, method='bilinear')

    logging.info('Computing velocity field')
    step = 8
    X, Y = np.meshgrid(np.arange(f1.shape[1]), np.arange(f1.shape[0]))
    vxs, vys = velocityfield(avecs, f1.shape, method='bilinear')

    vxs -= np.median(vxs.flat)
    vys -= np.median(vys.flat)

    figure()
    subplot(221)
    imshow(np.dstack((f1, f2, np.zeros_like(f1))))
    title('Overlaid frames')

    subplot(222)
    imshow(np.dstack((warped_f1, f2, np.zeros_like(f2))))
    title('Frame 1 warped to Frame 2 (image domain)')

    subplot(223)
    imshow(np.dstack((f1, f2, np.zeros_like(f2))))
    quiver(X[::step,::step], Y[::step,::step],
            -vxs[::step,::step]*f1.shape[1], -vys[::step,::step]*f1.shape[0],
            color='b', angles='xy', scale_units='xy', scale=1)
    title('Computed velocity field (median subtracted)')

    subplot(224)
    imshow(np.sqrt(vxs*vxs + vys*vys), interpolation='none', cmap=cm.hot)
    colorbar()
    title('Magnitude of computed velocity (median subtracted)')

register_frames(os.path.join(os.path.dirname(__file__), '..', 'tests', 'traffic.npz'))
register_frames(os.path.join(os.path.dirname(__file__), '..', 'tests', 'tennis.npz'))

show()
