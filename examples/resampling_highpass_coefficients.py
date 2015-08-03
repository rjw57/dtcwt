#!/usr/bin/env python
"""Show an example of how to re-sample high-pass DT-CWT coefficients.

"""
import os

import dtcwt
import dtcwt.compat
import dtcwt.sampling

# Use an off-screen backend for matplotlib
import matplotlib
matplotlib.use('agg')

# Import numpy and matplotlib's pyplot interface
import numpy as np
from matplotlib.pyplot import *

# Get a copy of the famous 'mandrill' image. In the default dtcwt tree, we ship
# one with the tests. The mandrill image is 512x512, floating point and has pixel
# values on the interval (0, 1].
mandrill = np.load(
    os.path.join(os.path.dirname(__file__), '..', 'tests', 'mandrill.npz')
)['mandrill']

# Chop a window out
mandrill = mandrill[224:288,224:288]

# We will try to re-scale mandrill by this amount and method
scale = 1.2
scale_method = 'lanczos'

def scale_direct(im):
    """Scale image directly."""
    return dtcwt.sampling.scale(im, (im.shape[0]*scale, im.shape[1]*scale), scale_method)

def scale_highpass(im):
    """Scale image assuming it to be wavelet highpass coefficients."""
    return dtcwt.sampling.scale_highpass(im, (im.shape[0]*scale, im.shape[1]*scale), scale_method)

# Rescale mandrill directly using default (Lanczos) sampling
mandrill_direct = scale_direct(mandrill)

# Transform mandrill
mandrill_l, mandrill_h = dtcwt.compat.dtwavexfm2(mandrill, nlevels=4)

# Re-scale each component and transform back. Do this both with and without
# shifting back to DC.
mandrill_l = scale_direct(mandrill_l)
mandrill_h_a, mandrill_h_b = [], []

for h in mandrill_h:
    mandrill_h_a.append(scale_direct(h))
    mandrill_h_b.append(scale_highpass(h))

# Transform back
mandrill_a = dtcwt.compat.dtwaveifm2(mandrill_l, mandrill_h_a)
mandrill_b = dtcwt.compat.dtwaveifm2(mandrill_l, mandrill_h_b)

figure(figsize=(10,10))

subplot(2,2,1)
imshow(mandrill, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Original')

subplot(2,2,2)
imshow(mandrill_direct, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Directly up-sampled')

subplot(2,2,3)
imshow(mandrill_a, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Up-sampled in the wavelet domain')

subplot(2,2,4)
imshow(mandrill_b, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Up-sampled in the wavelet domain with shifting')

tight_layout()
savefig('resampling-example.png')
