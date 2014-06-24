"""
.. note::
  This module is experimental. It's API may change between versions.

This module implements function for DTCWT-based image registration as outlined in
`[1] <http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5936113>`_.
These functions are 2D-only for the moment.

"""
from __future__ import division, absolute_import

import itertools

from six.moves import xrange

import dtcwt
import dtcwt.numpy
import dtcwt.sampling
import dtcwt.utils
import numpy as np

__all__ = [
    'estimatereg',
    'velocityfield',
    'warp',
    'warptransform',
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

    # Measure horizontal phase gradients by taking the angle of
    # summed conjugate products across horizontal pairs.
    S = (sb1[:,1:] * np.conj(sb1[:,:-1]) + sb2[:,1:] * np.conj(sb2[:,:-1])) * np.exp(-1j * w[0])
    #dx = np.angle(dtcwt.sampling.sample(S, xs-0.5, ys, method='bilinear')) + w[0]
    dx = np.hstack((
        np.angle(S[:,:1]),
        np.angle(0.5 * (S[:,:-1] + S[:,1:])),
        np.angle(S[:,-1:])
    )) + w[0]

    # Measure vertical phase gradients by taking the angle of
    # summed conjugate products across vertical pairs.
    S = (sb1[1:,:] * np.conj(sb1[:-1,:]) + sb2[1:,:] * np.conj(sb2[:-1,:])) * np.exp(-1j * w[1])
    dy = np.vstack((
        np.angle(S[:1,:]),
        np.angle(0.5 * (S[:-1,:] + S[1:,:])),
        np.angle(S[-1:,:])
    )) + w[1]

    # Measure temporal phase differences between refh and prevh
    A = sb2 * np.conj(sb1)
    dt = np.angle(A)

    return dy, dx, dt

def _pow2(a):
    return a * a

def _pow3(a):
    return a * a * a

def confidence(sb1, sb2, epsilon=1e-6):
    """
    Compute the confidence measure of subbands *sb1* and *sb2* which should be
    2d arrays of identical shape. The confidence measure gives highest weight
    to pixels we expect to give good results.

    epsilon is a small constant intended to regularise the situation in which
    the case when the wavelet coefficients are very small (and so dominated
    by noise). It should be set to be comparable to the cube of the amplitude
    of the measurement noise.
    """

    if sb1.size != sb2.size:
        raise ValueError('Subbands should have identical size')

    numerator, denominator = 0.0, epsilon

    # Pad subbands
    us = np.concatenate((
        np.concatenate((sb1[  :1,:1], sb1[  :1,:], sb1[  :1,-1:]), axis=1),
        np.concatenate((sb1[  : ,:1], sb1        , sb1[  : ,-1:]), axis=1),
        np.concatenate((sb1[-1: ,:1], sb1[-1: ,:], sb1[-1: ,-1:]), axis=1),
    ), axis=0)
    vs = np.concatenate((
        np.concatenate((sb2[  :1,:1], sb2[  :1,:], sb2[  :1,-1:]), axis=1),
        np.concatenate((sb2[  : ,:1], sb2        , sb2[  : ,-1:]), axis=1),
        np.concatenate((sb2[-1: ,:1], sb2[-1: ,:], sb2[-1: ,-1:]), axis=1),
    ), axis=0)

    us3_abs, vs3_abs = _pow3(np.abs(us)), _pow3(np.abs(vs))
    prod_coeffs = np.conj(us) * vs

    # pixels at -1, -1
    region = (slice(0,-2), slice(0,-2))
    numerator += prod_coeffs[region]
    denominator += us3_abs[region] + vs3_abs[region]

    # pixels at +1, -1
    region = (slice(0,-2), slice(2,None))
    numerator += prod_coeffs[region]
    denominator += us3_abs[region] + vs3_abs[region]

    # pixels at -1, +1
    region = (slice(2,None), slice(0,-2))
    numerator += prod_coeffs[region]
    denominator += us3_abs[region] + vs3_abs[region]

    # pixels at +1, +1
    region = (slice(2,None), slice(2,None))
    numerator += prod_coeffs[region]
    denominator += us3_abs[region] + vs3_abs[region]

    return _pow2(np.abs(numerator)) / denominator

Q_TRIU_INDICES = list(zip(*np.triu_indices(6)))
Q_TRIU_FLAT_INDICES = np.ravel_multi_index(np.triu_indices(6), (6,6))

def qtildematrices(t_ref, t_target, levels):
    r"""
    Compute :math:`\tilde{Q}` matrices for given levels.

    :param t_ref: the transformed reference image
    :param t_target: the transformed target image
    :param levels: a sequence of indices specifying which levels to examine
    :returns: a tuple of :math:`\tilde{Q}` matrices for each index in *levels*

    Both *t_ref* and *t_target* should be
    :py:class:`dtcwt.Pyramid`-compatible objects.
    Indices in *levels* are 0-based.

    The returned matrices are NxMx27 where NxM is the shape of the
    corresponding level's highpass subbands.

    """
    # Extract highpasses
    Yh1 = t_ref.highpasses
    Yh2 = t_target.highpasses

    # A list of arrays of \tilde{Q} matrices for each level
    Qt_mats = []

    for level in levels:
        highpasses1, highpasses2 = Yh1[level], Yh2[level]
        xs, ys = np.meshgrid(np.arange(0,1,1/highpasses1.shape[1]),
                             np.arange(0,1,1/highpasses1.shape[0]))

        Qt_mat_sum = None
        for subband in xrange(highpasses1.shape[2]):
            C_d = confidence(highpasses1[:,:,subband], highpasses2[:,:,subband])
            dy, dx, dt = phasegradient(highpasses1[:,:,subband], highpasses2[:,:,subband],
                                            EXPECTED_SHIFTS[subband,:])

            dx *= highpasses1.shape[1]
            dy *= highpasses1.shape[0]

            # This is the equivalent of the following for each member of the array
            #  Kt_mat = np.array(((1, 0, s*x, 0, s*y, 0, 0), (0, 1, 0, s*x, 0, s*y, 0), (0,0,0,0,0,0,1)))
            #  c_vec = np.array((dx, dy, -dt))
            #  tmp = (Kt_mat.T).dot(c_vec)
            tmp = (
                dx, dy, xs*dx, xs*dy, ys*dx, ys*dy, -dt
            )

            # Calculate Qmatrix elements
            Qt = np.zeros(dx.shape[:2] + (27,))
            elem_idx = 0

            # Q sub-matrix
            for r, c in Q_TRIU_INDICES:
                Qt[:,:,elem_idx] = tmp[r] * tmp[c]
                elem_idx += 1

            # q sub-vector
            for r in xrange(6):
                Qt[:,:,elem_idx] = tmp[r] * tmp[6]
                elem_idx += 1

            # Include the confidence parameter
            Qt *= C_d.reshape(Qt.shape[:-1] + (1,))**2

            # Update Qt mats
            if Qt_mat_sum is None:
                Qt_mat_sum = Qt
            else:
                Qt_mat_sum += Qt

        Qt_mats.append(Qt_mat_sum)

    return Qt_mats

def solvetransform(Qtilde_vec):
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

    # Convert from 27-element vector into Q matrix and vector
    Q = np.zeros(Qtilde_vec.shape[:-1] + (6*6,))
    Q[..., Q_TRIU_FLAT_INDICES] = Qtilde_vec[...,:21]
    q = Qtilde_vec[...,-6:]
    Q = np.reshape(Q, Qtilde_vec.shape[:-1] + (6,6))

    # Want to find a = -Q^{-1} q => Qa = -q
    # The naive way would be: return -np.linalg.inv(Q).dot(q)
    # A less naive way would be: return np.linalg.solve(Q, -q)

    # NumPy >1.8 directly supports using the last two dimensions as a matrix. IF
    # we get a LinAlgError, we assume that we need to fall-back to a NumPy 1.7
    # approach which is *significantly* slower.
    try:
        rv = np.linalg.solve(Q, -q)
    except np.linalg.LinAlgError:
        # Try the slower fallback
        rv = np.zeros(Qtilde_vec.shape[:-1] + (6,))
        for idx in itertools.product(*list(xrange(s) for s in Qtilde_vec.shape[:-1])):
            rv[idx] = np.linalg.solve(Q[idx], -q[idx])

    return rv

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

def warptransform(t, avecs, levels, method=None):
    """
    Return a warped version of a transformed image acting only on specified levels.

    :param t: a transformed image
    :param avecs: an array of affine distortion parameters
    :param levels: a sequence of 0-based indices specifying which levels to act on

    *t* should be a
    :py:class:`dtcwt.Pyramid`-compatible instance.

    The *method* parameter is interpreted as in :py:func:`dtcwt.sampling.rescale` and
    is the sampling method used to resize *avecs* to *shape*.

    .. note::

        This function will clone the transform *t* but it is a shallow clone
        where possible. Only the levels specified in *levels* will be
        deep-copied and warped.

    """
    warped_highpasses = list(t.highpasses)

    # Warp specified levels
    for l in levels:
        warped_highpasses[l] = warphighpass(warped_highpasses[l], avecs, method=method)

    # Clone the transform
    return dtcwt.numpy.Pyramid(t.lowpass, tuple(warped_highpasses), t.scales)

def estimatereg(source, reference, regshape=None):
    """
    Estimate registration from which will map *source* to *reference*.

    :param source: transformed source image
    :param reference: transformed reference image

    The *reference* and *source* parameters should support the same API as
    :py:class:`dtcwt.Pyramid`.

    The local affine distortion is estimated at at 8x8 pixel scales.
    Return a NxMx6 array where the 6-element vector at (N,M) corresponds to the
    affine distortion parameters for the 8x8 block with index (N,M).

    Use the :py:func:`velocityfield` function to convert the return value from
    this function into a velocity field.

    """
    # Extract number of levels and shape of level 3 subband
    nlevels = len(source.highpasses)
    if regshape is None:
        avecs_shape = source.highpasses[3].shape[:2] + (6,)
    else:
        avecs_shape = tuple(regshape[:2]) + (6,)

    # Initialise matrix of 'a' vectors
    avecs = np.zeros(avecs_shape)

    # Compute initial global transform
    levels = list(x for x in xrange(nlevels-1, nlevels-3, -1) if x>=0)
    Qt_mats = list(
            np.sum(np.sum(x, axis=0), axis=0)
            for x in qtildematrices(source, reference, levels)
    )
    Qt = np.sum(Qt_mats, axis=0)

    a = solvetransform(Qt)
    for idx in xrange(a.shape[0]):
        avecs[:,:,idx] = a[idx]

    # Refine estimate
    for it in xrange(2 * (nlevels-3) - 1):
        s, e = nlevels, nlevels - 2 - (it+1)//2
        levels = list(x for x in xrange(s, e-1, -1) if x>=2 and x<nlevels)
        if len(levels) == 0:
            continue

        # Warp the levels we'll be looking at with the current best-guess transform
        warped = warptransform(source, avecs, levels, method='bilinear')

        # Rescale and sample all the Qtilde matrix results
        all_qts = qtildematrices(warped, reference, levels)
        if all_qts is None or len(all_qts) < 1:
            continue

        qts = np.zeros(avecs.shape[:2] + all_qts[0].shape[2:])
        for x in all_qts:
            qts += dtcwt.sampling.rescale(_boxfilter(x, 3), avecs.shape[:2], method='nearest')

        avecs += solvetransform(qts)

    return avecs

def velocityfield(avecs, shape, method=None):
    """
    Given the affine distortion parameters returned from :py:func:`estimatereg`, return
    a tuple of 2D arrays giving the x- and y- components of the velocity field. The
    shape of the velocity component field is *shape*. The velocities are measured in
    terms of normalised units where the image has width and height of unity.

    The *method* parameter is interpreted as in :py:func:`dtcwt.sampling.rescale` and
    is the sampling method used to resize *avecs* to *shape*.

    """
    h, w = avecs.shape[:2]
    pxs, pys = np.meshgrid(np.arange(0, w, dtype=np.float32) / w,
                           np.arange(0, h, dtype=np.float32) / h)

    vxs = avecs[:,:,0] + avecs[:,:,2] * pxs + avecs[:,:,4] * pys
    vys = avecs[:,:,1] + avecs[:,:,3] * pxs + avecs[:,:,5] * pys

    vxs = dtcwt.sampling.rescale(vxs, shape, method=method)
    vys = dtcwt.sampling.rescale(vys, shape, method=method)

    return vxs, vys

def warphighpass(Yh, avecs, method=None):
    """
    A convenience function to warp a highpass subband image according to the
    velocity field implied by *avecs*.

    This function correctly 'de-rotates' the highpass image before sampling and
    're-rotates' afterwards.

    """
    X, Y = np.meshgrid(np.arange(Yh.shape[1], dtype=np.float32)/Yh.shape[1],
                       np.arange(Yh.shape[0], dtype=np.float32)/Yh.shape[0])
    vxs, vys = velocityfield(avecs, Yh.shape, method=method)
    return normsamplehighpass(Yh, X+vxs, Y+vys, method=method)

def warp(I, avecs, method=None):
    """
    A convenience function to warp an image according to the velocity field
    implied by *avecs*.

    """
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
