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
from dtcwt.backend import backend_numpy
import dtcwt.sampling
import dtcwt.utils
import numpy as np

__all__ = [
    'EXPECTED_SHIFTS',

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
    :py:class:`dtcwt.backend.base.TransformDomainSignal`-compatible objects.
    Indices in *levels* are 0-based.

    The returned matrices are NxMx27 where NxM is the shape of the
    corresponding level's highpass subbands.

    """
    # Extract subbands
    Yh1 = t_ref.subbands
    Yh2 = t_target.subbands

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

            dx *= subbands1.shape[1]
            dy *= subbands1.shape[0]

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
    #
    # Recall that, given Q is symmetric, we can decompose it at Q = U L U^T with
    # diagonal L and orthogonal U. Hence:
    #
    #   U L U^T a = -q => a = - U L^-1 U^T q
    #
    # An even better way is thus to use the specialised eigenvector
    # calculation for symmetric matrices:

    # NumPy >1.8 directly supports using the last two dimensions as a matrix. IF
    # we get a LinAlgError, we assume that we need to fall-back to a NumPy 1.7
    # approach which is *significantly* slower.
    try:
        l, U = np.linalg.eigh(Q, 'U')
    except np.linalg.LinAlgError:
        # Try the slower fallback
        l = np.zeros(Qtilde_vec.shape[:-1] + (6,))
        U = np.zeros(Qtilde_vec.shape[:-1] + (6,6))
        for idx in itertools.product(*list(xrange(s) for s in Qtilde_vec.shape[:-1])):
            l[idx], U[idx] = np.linalg.eigh(Q[idx], 'U')

    # Now we have some issue here. If Qtilde_vec is a straightforward vector
    # then we can just return U.dot((U.T.dot(-q))/l). However if Qtilde_vec is
    # an array of vectors then the straightforward dot product won't work. We
    # want to perform matrix multiplication on stacked matrices.
    return dtcwt.utils.stacked_2d_matrix_vector_prod(
        U, dtcwt.utils.stacked_2d_vector_matrix_prod(-q, U) / l
    )

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
    :pyclass:`dtcwt.backend.base.TransformDomainSignal`-compatible instance.

    The *method* parameter is interpreted as in :py:func:`dtcwt.sampling.rescale` and
    is the sampling method used to resize *avecs* to *shape*.

    .. note::

        This function will clone the transform *t* but it is a shallow clone
        where possible. Only the levels specified in *levels* will be
        deep-copied and warped.

    """
    warped_subbands = list(t.subbands)

    # Warp specified levels
    for l in levels:
        warped_subbands[l] = warphighpass(warped_subbands[l], avecs, method=method)

    # Clone the transform
    return backend_numpy.TransformDomainSignal(t.lowpass, tuple(warped_subbands), t.scales)

def estimatereg(reference, target):
    """
    Estimate registration from reference image to target.

    :param reference: transformed reference image
    :param target: transformed target image

    The *reference* and *transform* parameters should support the same API as
    :py:class:`dtcwt.backend.base.TransformDomainSignal`.

    The local affine distortion is estimated at at 8x8 pixel scales.
    Return a NxMx6 array where the 6-element vector at (N,M) corresponds to the
    affine distortion parameters for the 8x8 block with index (N,M).

    Use the :py:func:`velocityfield` function to convert the return value from
    this function into a velocity field.

    """
    # Extract number of levels and shape of level 3 subband
    nlevels = len(reference.subbands)
    avecs_shape = reference.subbands[2].shape[:2] + (6,)

    # Initialise matrix of 'a' vectors
    avecs = np.zeros(avecs_shape)

    # Compute initial global transform
    levels = list(x for x in xrange(nlevels-1, nlevels-3, -1) if x>=0)
    Qt_mats = list(np.sum(np.sum(x, axis=0), axis=0) for x in qtildematrices(reference, target, levels))
    Qt = np.sum(Qt_mats, axis=0)

    a = solvetransform(Qt)
    for idx in xrange(a.shape[0]):
        avecs[:,:,idx] = a[idx]

    # Refine estimate
    for it in xrange(2 * (nlevels-3) - 1):
        s, e = nlevels - 2, nlevels - 2 - (it+1)//2
        levels = list(x for x in xrange(s, e-1, -1) if x>=2 and x<nlevels)
        if len(levels) == 0:
            continue

        # Warp the levels we'll be looking at with the current best-guess transform
        warped = warptransform(reference, avecs, levels, method='bilinear')

        qts = np.sum(list(dtcwt.sampling.rescale(_boxfilter(x, 3), avecs.shape[:2], method='bilinear')
                          for x in qtildematrices(warped, target, levels)),
                     axis=0)

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
