"""
.. note::
  This module is experimental. It's API may change between versions.

Functions for DTCWT-based image registration as outlined in
`[1] <http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5936113>`_.
These functions are 2D-only for the moment.

"""

from __future__ import division

import itertools

from six.moves import xrange

import dtcwt
import dtcwt.sampling
import dtcwt.utils
import numpy as np

__all__ = [
    'EXPECTED_SHIFTS',

    'affinevelocityfield',
    'affinewarp',
    'affinewarphighpass',
    'confidence',
    'estimateflow',
    'normsample',
    'normsamplehighpass',
    'phasegradient',
    'qtildematrices',
    'solvetransform',
    'velocityfield',
    'warp',
    'warphighpass',
]

# Horizontal and vertical expected phase shifts for each subband
EXPECTED_SHIFTS = np.array(((-1,-3),(-3,-3),(-3,-1),(-3,1),(-3,3),(-1,3))) * np.pi / 2.15

def phasegradient(sb1, sb2, w=None):
    """
    Compute the dx, dy and dt phase gradients for subbands *sb1* and *sb2*.
    Each subband should be an NxM matrix of complex values and should be the
    same size.

    *w* should be a pair giving the expected phase shift for horizontal and vertical
    directions. This expected phase shift is used to 'de-rotate' the phase gradients
    before computing the angle and is added afterwards. If not specified, 0 is
    used. This is likely to give poor results.

    Returns 3 NxM matrices giving the d/dy, d/dx and d/dt phase values
    respectively at position (x,y).

    """
    if w is None:
        w = (0,0)

    if sb1.size != sb2.size:
        raise ValueError('Subbands should have identical size')

    xs, ys = np.meshgrid(np.arange(sb1.shape[1]), np.arange(sb1.shape[0]))

    # Measure horizontal phase gradients by taking the angle of
    # summed conjugate products across horizontal pairs.
    A = sb1[:,1:] * np.conj(sb1[:,:-1])
    B = sb2[:,1:] * np.conj(sb2[:,:-1])
    dx = np.angle(dtcwt.sampling.sample((A+B) * np.exp(-1j * w[0]), xs-0.5, ys, method='bilinear')) + w[0]

    # Measure vertical phase gradients by taking the angle of
    # summed conjugate products across vertical pairs.
    A = sb1[1:,:] * np.conj(sb1[:-1,:])
    B = sb2[1:,:] * np.conj(sb2[:-1,:])
    dy = np.angle(dtcwt.sampling.sample((A+B) * np.exp(-1j * w[1]), xs, ys-0.5, method='bilinear')) + w[1]

    # Measure temporal phase differences between refh and prevh
    A = sb2 * np.conj(sb1)
    dt = np.angle(A)

    return dy, dx, dt

def confidence(sb1, sb2):
    """
    Compute the confidence measure of subbands *sb1* and *sb2* which should be
    2d arrays of identical shape. The confidence measure gives highest weight
    to pixels we expect to give good results.

    """
    if sb1.size != sb2.size:
        raise ValueError('Subbands should have identical size')

    xs, ys = np.meshgrid(np.arange(sb1.shape[1]), np.arange(sb1.shape[0]))

    numerator, denominator = 0, 1e-6

    for dx, dy in ((-1,-1), (-1,1), (1,-1), (1,1)):
        us = dtcwt.sampling.sample(sb1, xs+dx, ys+dy, method='nearest')
        vs = dtcwt.sampling.sample(sb2, xs+dx, ys+dy, method='nearest')

        numerator += np.power(np.abs(np.conj(us) * vs), 2)
        denominator += np.power(np.abs(us), 3) + np.power(np.abs(vs), 3)

    return numerator / denominator

def qtildematrices(Yh1, Yh2, levels):
    r"""
    Compute :math:`\tilde{Q}` matrices for given levels.

    """
    def _Qt_mat(x, y, dx, dy, dt):
        # The computation that this function performs is:
        #
        # Kt_mat = np.array(((1, 0, s*x, 0, s*y, 0, 0), (0, 1, 0, s*x, 0, s*y, 0), (0,0,0,0,0,0,1)))
        # c_vec = np.array((dx, dy, -dt))
        # a = (Kt_mat.T).dot(c_vec)

        a = np.array((
            dx, dy, x*dx, x*dy, y*dx, y*dy, -dt
        ))

        return np.outer(a, a)

    _Qt_mats = np.vectorize(_Qt_mat, otypes='O')

    # A list of arrays of \tilde{Q} matrices for each level
    Qt_mats = []

    for level in levels:
        subbands1, subbands2 = Yh1[level], Yh2[level]
        xs, ys = np.meshgrid(np.arange(0,1,1/subbands1.shape[1]),
                             np.arange(0,1,1/subbands1.shape[0]))

        Qt_mat_sum = None
        for subband in xrange(subbands1.shape[2]):
            C_d = confidence(subbands1[:,:,subband], subbands2[:,:,subband])
            dy, dx, dt = phasegradient(subbands1[:,:,subband], subbands2[:,:,subband],
                                            EXPECTED_SHIFTS[subband,:])

            Qt = _Qt_mats(xs, ys, dx*subbands1.shape[1], dy*subbands1.shape[0], dt)

            # Update Qt mats
            if Qt_mat_sum is None:
                Qt_mat_sum = Qt
            else:
                Qt_mat_sum += Qt

        Qt_mats.append(Qt_mat_sum)

    return Qt_mats

def solvetransform(Qtilde):
    r"""
    Solve for affine transform parameter vector :math:`a` from :math:`\mathbf{\tilde{Q}}`
    matrix. decomposes :math:`\mathbf{\tilde{Q}}` as

    .. math::
        \tilde{Q} = \begin{bmatrix}
            \mathbf{Q}   & \mathbf{q}   \\
            \mathbf{q}^T & q_0
        \end{bmatrix}

    Returns :math:`\mathbf{a} = -\mathbf{Q}^{-1} \mathbf{q}`.
    """
    Q = Qtilde[:-1,:-1]
    q = Qtilde[-1,:-1]
    return -np.linalg.inv(Q).dot(q)

def normsamplehighpass(Yh, xs, ys, method=None):
    """
    Given a NxMx6 array of subband responses, sample from co-ordinates *xs* and
    *ys*.

    .. note::
      The sample co-ordinates are in *normalised* co-ordinates such that the
      image width and height are both unity.

    """
    return dtcwt.sampling.sample_highpass(Yh, xs*Yh.shape[1], ys*Yh.shape[0], method=method)

def normsample(Yh, xs, ys, method=None):
    """
    Given a NxM image sample from co-ordinates *xs* and *ys*.

    .. note::
      The sample co-ordinates are in *normalised* co-ordinates such that the
      image width and height are both unity.

    """
    return dtcwt.sampling.sample(Yh, xs*Yh.shape[1], ys*Yh.shape[0], method=method)

def affinevelocityfield(a, xs, ys):
    r"""
    Evaluate the velocity field parametrised by *a* at *xs* and *ys*.

    Return a pair *vx*, *vy* giving the x- and y-component of the velocity.

    The velocity vector is given by :math:`\mathbf{v} = \mathbf{K} \mathbf{a}` where

    .. math::
      \mathbf{K} = \begin{bmatrix}
            1 & 0 & x & 0 & y & 0  \\
            0 & 1 & 0 & x & 0 & y
          \end{bmatrix}

    """
    return (a[0] + a[2]*xs + a[4]*ys), (a[1] + a[3]*xs + a[5]*ys)

def affinewarphighpass(Yh, a, method=None):
    r"""
    Given a NxMx6 array of subband responses, warp it according to affine transform
    parametrised by the vector *a* s.t. the pixel at (x,y) samples from (x',y') where

    .. math::
      [x', y']^T = \mathbf{T} \ [1, x, y]^T

    and

    .. math::
      \mathbf{T} = \begin{bmatrix}
            a_0 & a_2 & a_4  \\
            a_1 & a_3 & a_5
          \end{bmatrix}

    .. note::
      The sample co-ordinates are in *normalised* co-ordinates such that the
      image width and height are both unity.

    """
    xs, ys = np.meshgrid(np.arange(0,1,1/Yh.shape[1]), np.arange(0,1,1/Yh.shape[0]))
    vxs, vys = affinevelocityfield(a, xs, ys)
    return normsamplehighpass(Yh, xs+vxs, ys+vys, method=method)

def affinewarp(Yh, a, method=None):
    r"""
    Given a NxM image, warp it according to affine transform parametrised by
    the vector *a* s.t. the pixel at (x,y) samples from (x',y') where

    .. math::
      [x', y']^T = \mathbf{T} \ [1, x, y]^T

    and

    .. math::
      \mathbf{T} = \begin{bmatrix}
            a_0 & a_2 & a_4  \\
            a_1 & a_3 & a_5
          \end{bmatrix}

    .. note::
      The sample co-ordinates are in *normalised* co-ordinates such that the
      image width and height are both unity.

    """
    xs, ys = np.meshgrid(np.arange(0,1,1/Yh.shape[1]), np.arange(0,1,1/Yh.shape[0]))
    vxs, vys = affinevelocityfield(a, xs, ys)
    return normsample(Yh, xs+vxs, ys+vys, method=method)

def estimateflow(reference_h, target_h):
    # Make a copy of reference_h for warping
    warped_h = list(reference_h)

    # Initialise matrix of 'a' vectors
    avecs = np.zeros((warped_h[2].shape[0], warped_h[2].shape[1], 6))

    # Compute initial global transform
    levels = list(x for x in xrange(len(warped_h)-1, len(warped_h)-3, -1) if x>=0)
    Qt_mats = list(x.sum() for x in qtildematrices(warped_h, target_h, levels))
    Qt = np.sum(Qt_mats, axis=0)

    a = solvetransform(Qt)
    for r, c in itertools.product(xrange(avecs.shape[0]), xrange(avecs.shape[1])):
        avecs[r,c] += a

    for l in levels:
        warped_h[l] = warphighpass(reference_h[l], avecs, method='bilinear')

    # Refine estimate
    for it in xrange(2 * (len(warped_h)-3) - 1):
        s, e = len(warped_h) - 2, len(warped_h) - 2 - (it+1)//2
        levels = list(x for x in xrange(s, e-1, -1) if x>=2 and x<len(warped_h))
        if len(levels) == 0:
            continue

        qts = np.sum(list(dtcwt.sampling.rescale(_boxfilter(x, 3), avecs.shape[:2], method='bilinear')
                          for x in qtildematrices(warped_h, target_h, levels)),
                     axis=0)

        for r, c in itertools.product(xrange(avecs.shape[0]), xrange(avecs.shape[1])):
            # if np.abs(np.linalg.det(qts[r,c][:-1,:-1])) > np.abs(qts[r,c][-1,-1]):
            try:
                avecs[r,c] += solvetransform(qts[r,c])
            except np.linalg.LinError:
                # It's OK for refinement to generate singular matrices
                pass

        for l in levels:
            warped_h[l] = warphighpass(reference_h[l], avecs, method='bilinear')

    return avecs

def velocityfield(avecs, shape, method=None):
    h, w = shape[:2]
    pxs, pys = np.meshgrid(np.arange(0, w, dtype=np.float32) / w,
                           np.arange(0, h, dtype=np.float32) / h)

    avecs = dtcwt.sampling.rescale(avecs, shape, method='bilinear')
    vxs = avecs[:,:,0] + avecs[:,:,2] * pxs + avecs[:,:,4] * pys
    vys = avecs[:,:,1] + avecs[:,:,3] * pxs + avecs[:,:,5] * pys

    return vxs, vys

def warphighpass(Yh, avecs, method=None):
    X, Y = np.meshgrid(np.arange(Yh.shape[1], dtype=np.float32)/Yh.shape[1],
                       np.arange(Yh.shape[0], dtype=np.float32)/Yh.shape[0])
    vxs, vys = velocityfield(avecs, Yh.shape, method=method)
    return normsamplehighpass(Yh, X+vxs, Y+vys, method=method)

def warp(I, avecs, method=None):
    X, Y = np.meshgrid(np.arange(I.shape[1], dtype=np.float32)/I.shape[1],
                       np.arange(I.shape[0], dtype=np.float32)/I.shape[0])
    vxs, vys = velocityfield(avecs, I.shape, method=method)
    return normsample(I, X+vxs, Y+vys, method=method)

def _boxfilter(X, kernel_size):
    """
    INTERNAL

    A simple box filter implementation.

    """
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be odd')

    for axis_idx in xrange(2):
        slices = [slice(None),] * len(X.shape)
        out = X

        for delta in xrange(1, 1+(kernel_size-1)//2):
            slices[axis_idx] = dtcwt.utils.reflect(
                    np.arange(X.shape[axis_idx]) + delta, -0.5, X.shape[axis_idx]-0.5)
            out = out + X[slices]
            slices[axis_idx] = dtcwt.utils.reflect(
                    np.arange(X.shape[axis_idx]) - delta, -0.5, X.shape[axis_idx]-0.5)
            out = out + X[slices]

        X = out / kernel_size

    return X
