#!/usr/bin/env python
"""Show an example of how to re-sample high-pass DT-CWT coefficients.

"""
import os

import dtcwt
import dtcwt.sampling

# Use an off-screen backend for matplotlib
import matplotlib
matplotlib.use('agg')

# Import numpy and matplotlib's pyplot interface
import numpy as np
from matplotlib.pyplot import *

# Get a copy of the famous 'lena' image. In the default dtcwt tree, we ship
# one with the tests. The lena image is 512x512, floating point and has pixel
# values on the interval (0, 1].
lena = np.load(
    os.path.join(os.path.dirname(__file__), '..', 'tests', 'lena.npz')
)['lena']

# Chop a window out
lena = lena[224:288,224:288]

# We will try to re-scale lena by this amount and method
scale = 1.2
scale_method = 'lanczos'

def scale_direct(im):
    """Scale image directly."""
    return dtcwt.sampling.scale(im, (im.shape[0]*scale, im.shape[1]*scale), scale_method)

def scale_highpass(im):
    """Scale image assuming it to be wavelet highpass coefficients."""
    return dtcwt.sampling.scale_highpass(im, (im.shape[0]*scale, im.shape[1]*scale), scale_method)

# Rescale lena directly using default (Lanczos) sampling
lena_direct = scale_direct(lena)

# Transform lena
lena_l, lena_h = dtcwt.dtwavexfm2(lena, nlevels=4)

# Re-scale each component and transform back. Do this both with and without
# shifting back to DC.
lena_l = scale_direct(lena_l)
lena_h_a, lena_h_b = [], []

for h in lena_h:
    lena_h_a.append(scale_direct(h))
    lena_h_b.append(scale_highpass(h))

# Transform back
lena_a = dtcwt.dtwaveifm2(lena_l, lena_h_a)
lena_b = dtcwt.dtwaveifm2(lena_l, lena_h_b)

figure(figsize=(10,10))

subplot(2,2,1)
imshow(lena, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Original')

subplot(2,2,2)
imshow(lena_direct, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Directly up-sampled')

subplot(2,2,3)
imshow(lena_a, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Up-sampled in the wavelet domain')

subplot(2,2,4)
imshow(lena_b, cmap=cm.gray, clim=(0,1), interpolation='none')
axis('off')
title('Up-sampled in the wavelet domain with shifting')

tight_layout()
savefig('resampling-example.png')
