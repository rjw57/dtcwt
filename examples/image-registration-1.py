#!/usr/bin/env python
"""
An example of image registration via the DTCWT.

This script demonstrates some methods for image registration using the DTCWT.

"""

from __future__ import division, print_function

import os
import logging

from matplotlib.pyplot import *
import numpy as np

import dtcwt
import dtcwt.sampling
from dtcwt.registration import *

logging.basicConfig(level=logging.INFO)

# Load test image
logging.info('Loading Lena image')
f1 = np.load(os.path.join(os.path.dirname(__file__), '..', 'tests', 'lena.npz'))['lena']

# Compute transformed image. Note that we use normalised image co-ords so that the image is on the region [0,1) x [0,1).

logging.info('Creating transformed image')
xs, ys = np.meshgrid(np.arange(f1.shape[1], dtype=np.float32)/f1.shape[1],
                     np.arange(f1.shape[0], dtype=np.float32)/f1.shape[0])

def trans(dx, dy):
    return np.array([[1,0,dx], [0,1,dy], [0,0,1]])

def rot(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[c,s,0], [-s,c,0], [0,0,1]])

theta = np.deg2rad(10)
dx, dy = 20/f1.shape[1], -30/f1.shape[0]

T = trans(0.5+dx, 0.5+dy).dot(rot(theta)).dot(trans(-0.5,-0.5))

xs_prime = T[0,0]*xs + T[0,1]*ys + T[0,2]
ys_prime = T[1,0]*xs + T[1,1]*ys + T[1,2]

f2 = dtcwt.sampling.sample(f1, xs_prime*f1.shape[1], ys_prime*f1.shape[0], method='bilinear')

# Also compute a ground truth velocity field. If a point $(x,y)$ is transformed
# to $(x', y')$ where $[x', y']^T = T \, [x, y]^T$ then:
#
# $$ \Delta_x = (x'-x) = T_{11}x + T_{12}y + T_{13} - x $$
# $$ \Delta_y = (y'-y) = T_{21}x + T_{22}y + T_{23} - y $$

logging.info('Computing ground truth velocity field')
pxs, pys = np.arange(0, f1.shape[1], 10, dtype=np.float32), np.arange(0, f1.shape[0], 10, dtype=np.float32)
X, Y = np.meshgrid(pxs, pys)
X /= f1.shape[1]
Y /= f1.shape[0]
gt_vxs = T[0,2] + T[0,0]*X + T[0,1]*Y - X
gt_vys = T[1,2] + T[1,0]*X + T[1,1]*Y - Y

# Take the DTCWT of both frames.
logging.info('Taking DTCWT')
nlevels = 8
Yl1, Yh1 = dtcwt.dtwavexfm2(f1, nlevels=nlevels)
Yl2, Yh2 = dtcwt.dtwavexfm2(f2, nlevels=nlevels)

# Iteratively solve for transform

# Start with a set of warped high-pass subbands which
# are a shallow copy of Yh1
Yh3 = list(Yh1)

# Initial velocity field is zero
a = np.zeros(6, dtype=np.float32)

for iteration in xrange(4*(len(Yh1)-5)):
    startlevel = len(Yh1)-2-iteration//4
    levels = [startlevel, startlevel+1]

    logging.info('Refining with levels: %s' % (levels,))

    # Warp Yh1 for each level we're looking at in this iteration
    for level in levels:
        Yh3[level] = affinewarphighpass(Yh1[level], a, method='bilinear')

    Qt_mats = qtildematrices(Yh3, Yh2, levels)
    Qt = np.sum(list(x.sum() for x in Qt_mats), axis=0)
    a += solvetransform(Qt)

logging.info('Computing velocity field')

# Compute and plot the reconstructed velocity field and compare it to the ground truth.
vxs = a[0] + a[2]*X + a[4]*Y
vys = a[1] + a[3]*X + a[5]*Y

figure()

subplot(121)
imshow(f1, cmap=cm.gray)
quiver(pxs, pys, gt_vxs, gt_vys, color='r', angles='xy')
title('Ground truth velocity field')

subplot(122)
imshow(f1, cmap=cm.gray)
quiver(pxs, pys, vxs, vys, color='r', angles='xy')
title('Computed velocity field')

figure()
subplot(221)
imshow(vxs)
title('Computed velocity x-component')
colorbar()

subplot(222)
imshow(gt_vxs)
title('Ground truth velocity x-component')
colorbar()

subplot(223)
imshow(vys)
title('Computed velocity y-component')
colorbar()

subplot(224)
imshow(gt_vys)
title('Ground truth velocity y-component')
colorbar()

# Compute a high-quality warping using Lanczos re-sampling
logging.info('Computing high-quality warped image')
warped_direct = affinewarp(f1, a, method='lanczos')

figure()
subplot(121)
imshow(f2, cmap=cm.gray, clim=(0,1))
title('Frame 2')

subplot(122)
imshow(warped_direct, cmap=cm.gray, clim=(0,1))
title('Frame 1 warped to Frame 2')

show()
