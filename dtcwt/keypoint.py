from __future__ import absolute_import

import numpy as np

from dtcwt.sampling import upsample_highpass, upsample

__all__ = [ 'find_keypoints' ]

def find_keypoints(highpass_highpasses, method=None,
        alpha=1.0, beta=0.4, kappa=1.0/6.0,
        threshold=None, max_points=None,
        upsample_keypoint_energy=None, upsample_highpasses=None,
        refine_positions=True, skip_levels=1):
    """
    :param highpass_highpasses: (NxMx6) matrix of highpass subband images
    :param method: *(optional)* string specifying which keypoint energy method to use
    :param alpha: *(optional)* scale parameter for ``'fauqueur'`` method
    :param beta: *(optional)* shape parameter for ``'fauqueur'`` method
    :param kappa: *(optiona)* suppression parameter for ``'kingsbury'`` method
    :param threshold: *(optional)* minimum keypoint energy of returned keypoints
    :param max_points: *(optional)* maximum number of keypoints to return
    :param upsample_keypoint_energy: is non-None, a string specifying a method used to upscale the keypoint energy map before finding keypoints
    :param upsample_subands: is non-None, a string specifying a method used to upscale the subband image before finding keypoints
    :param refine_positions: *(optional)* should the keypoint positions be refined to sub-pixel accuracy
    :param skip_levels: *(optional)* number of levels of the transform to ignore before looking for keypoints

    :returns: (Px4) array of P keypoints in image co-ordinates

    .. warning::

        The interface and behaviour of this function is the subject of an open
        research project. It is provided in this release as a preview of
        forthcoming functionality but it is subject to change between releases.

    The rows of the returned keypoint array give the x co-ordinate, y
    co-ordinate, scale and keypoint energy. The rows are sorted in order of
    decreasing keypoint energy.

    If *refine_positions* is ``True`` then the positions (and energy) of the
    keypoints will be refined to sub-pixel accuracy by fitting a quadratic
    patch. If *refine_positions* is ``False`` then the keypoint locations will
    be those corresponding directly to pixel-wise maxima of the subband images.

    The *max_points* and *threshold* parameters are cumulative: if both are
    specified then the *max_points* greatest energy keypoints with energy
    greater than *threshold* will be returned.

    Usually the keypoint energies returned from the finest scale level are
    dominated by noise and so one usually wants to specify *skip_levels* to be
    1 or 2. If *skip_levels* is 0 then all levels will be used to compute
    keypoint energy.

    The *upsample_highpasses* and *upsample_keypoint_energy* parameters are used
    to control whether the individual subband coefficients and/org the keypoint
    energy map are upscaled by 2 before finding keypoints. If these parameters
    are None then no corresponding upscaling is performed. If non-None they
    specify the upscale method as outlined in
    :py:func:`dtcwt.sampling.upsample`.

    If *method* is ``None``, the default ``'fauqueur'`` method is used.

    =========== ======================================= ======================
    Name        Description                             Parameters used
    =========== ======================================= ======================
    fauqueur    Geometric mean of absolute values[1]    *alpha*, *beta*
    bendale     Minimum absolute value[2]               none
    kingsbury   Cross-product of orthogonal highpasses    *kappa*
    =========== ======================================= ======================

    [1] Julien Fauqueur, Nick Kingsbury, and Ryan Anderson. *Multiscale
    Keypoint Detection using the Dual-Tree Complex Wavelet Transform*. 2006
    International Conference on Image Processing, pages 1625-1628, October
    2006. ISSN 1522-4880. doi: 10.1109/ICIP.2006.312656.
    http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=4106857.

    [2] Pashmina Bendale, Bill Triggs, and Nick Kingsbury. *Multiscale Keypoint
    Analysis based on Complex Wavelets*. In British Machine Vision Con-ference
    (BMVC), 2010.
    http://www-sigproc.eng.cam.ac.uk/~pb397/publications/BTK_BMVC_2010_abstract.pdf.
    """
    # Set default method
    if method is None:
        method = 'fauqueur'

    # Skip levels
    highpass_highpasses = highpass_highpasses[skip_levels:]

    # Compute contribution to scale from upsampling
    upsample_scale = 1
    if upsample_highpasses is not None:
        upsample_scale <<= 1
    if upsample_keypoint_energy is not None:
        upsample_scale <<= 1

    # Find keypoint energy map for each level
    kp_energies = []
    for subband in highpass_highpasses:
        if upsample_highpasses is not None:
            subband = upsample_highpass(subband, upsample_highpasses)

        if method == 'fauqueur':
            kp_energies.append(_keypoint_energy_fauqueur(subband, alpha, beta))
        elif method == 'bendale':
            kp_energies.append(_keypoint_energy_bendale(subband))
        elif method == 'kingsbury':
            kp_energies.append(_keypoint_energy_kingsbury(subband, kappa))
        elif method == 'gale':
            kp_energies.append(_keypoint_energy_gale(subband))
        else:
            raise ValueError('Unknown method: {0}'.format(method))

        if upsample_keypoint_energy is not None:
            kp_energies[-1] = upsample(kp_energies[-1], upsample_keypoint_energy)

    # Find keypoints for each level
    kps = None
    for level_idx, kp_energy in enumerate(kp_energies):
        kp_scale = 2**(level_idx+1+skip_levels) / float(upsample_scale)
        kp_rows, kp_cols, kp_energies = _kp_energy_maxima(kp_energy, threshold=threshold, refine=refine_positions)

        # Scaling is a bit non-trivial. If the subband has pixel coords in range {0, .., M-1} then it has extent
        # (-0.5, M-0.5]. If we need to scale the pixel size by kp_scale then the final image will have extent
        # (-0.5, kp_scale*M-0.5]. So we need a linear function which maps -0.5 -> -0.5 and M-0.5 -> kp_scale*M-0.5
        # such a function is x -> kp_scale * (x+0.5) - 0.5

        level_kps = np.array((
            (kp_cols+0.5)*kp_scale-0.5, (kp_rows+0.5)*kp_scale-0.5,
            kp_scale*np.ones(kp_cols.shape[0]), kp_energies)).T

        if kps is None:
            kps = level_kps
        else:
            kps = np.vstack((kps, level_kps))

    # Sort keypoints
    sorted_indices = np.argsort(kps[:, 3])
    kps = kps[sorted_indices[::-1],:]

    # Truncate if necessary
    if max_points is not None:
        kps = kps[:max_points]

    # Return keypoints
    return kps

def _keypoint_energy_fauqueur(subband, alpha, beta):
    return alpha * np.power(np.maximum(0, np.product(np.abs(subband), axis=2)), beta)

def _keypoint_energy_bendale(subband):
    return np.min(np.abs(subband), axis=2)

def _keypoint_energy_kingsbury(subband, kappa=1.0/6.0, epsilon=1e-8):
    abs_Y = np.abs(subband)
    A = np.sqrt(np.sum(abs_Y*abs_Y, axis=2))
    B = np.sum(abs_Y[:,:,:3] * abs_Y[:,:,3:], axis=2)

    # The max(0, ...) is not part of the original energy calculation but we use
    # it to avoid finding false maxima in no-threshold cases.
    return np.maximum(0, B/np.maximum(epsilon, A) - kappa*A)

def _keypoint_energy_gale(subband):
    raise NotImplementedError('not implemented yet')

def _nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def _kp_energy_maxima(X, threshold=None, refine=True):
    # If no threshold is provided, choose one which all keypoints will pass
    if threshold is None:
        threshold = X.min() - 1

    # Compute local maximum image
    maxima = np.ones_like(X) * threshold
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            maxima[1:-2,1:-2] = np.maximum(maxima[1:-2,1:-2], X[1+dy:X.shape[0]-2+dy, 1+dx:X.shape[1]-2+dx])

    # This will be used to store the values of local maxima
    lm_values = []

    # This will be used to store the refined positions
    lm_refined_rows, lm_refined_cols = [], []
            
    # Find local maxima
    lm_rows, lm_cols = np.nonzero(maxima == X)
    
    if refine:
        # Taylor series of I(x) around x_0 is I(x) ~= I(x_o) + dI/dx x + dI^2/dx^2 x^2 + ...
        # maximum is at differential is zero or:
        # 0 = dI/dx + 2 dI^2/dx^2 x => x = -dI/dx * (2*dI^2/dx^2)^-1

        # Form the various gradient images for X
        dXdy, dXdx = np.gradient(X)
        dX2dxdy, dX2dx2 = np.gradient(dXdx)
        dX2dy2, dX2dydx = np.gradient(dXdy)
        
        a_im = np.dstack((
           dX2dx2, dX2dy2, dX2dxdy, dXdx, dXdy, X,
        ))
    
    # Calculate a vectors for each neighbourhood
    for r, c in zip(lm_rows, lm_cols):
        if refine:
            a = a_im[r,c,:]
            A = np.array(((2*a[0], a[2], a[3]), (a[2], 2*a[1], a[4])))
            v = _nullspace(A)[:,0]
            v /= v[2]
            
            # Only accept fittings where maximum is within half of a pixel of
            # the maximum pixel's centre.
            if np.any(np.abs(v[:2]) > 0.5):
                continue
        
            x, y = v[:2]
            lm_values.append(a[0]*x*x + a[1]*y*y + a[2]*x*y + a[3]*x + a[4]*y + a[5])
        else:
            x, y = 0, 0
            lm_values.append(X[r,c])

        lm_refined_rows.append(r+y)
        lm_refined_cols.append(c+x)
    
    return np.array(lm_refined_rows), np.array(lm_refined_cols), np.array(lm_values)
