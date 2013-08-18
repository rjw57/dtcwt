"""Rescaling and re-sampling high- and low-pass subbands.

"""

__all__ = (
    'sample', 'sample_highpass', 'scale', 'scale_highpass'
)

import numpy as np

_W0 = -3*np.pi/2.0
_W1 = -np.pi/2.0

DEFAULT_SAMPLE_METHOD = 'lanczos'

#: The expected phase advances in the x-direction for each subband of the 2D transform
DTHETA_DX_2D = np.array((_W1, _W0, _W0, _W0, _W0, _W1))

#: The expected phase advances in the y-direction for each subband of the 2D transform
DTHETA_DY_2D = np.array((_W0, _W0, _W1, -_W1, -_W0, -_W0))

def _sample_clipped(im, xs, ys):
    """Truncated and clipped sampling."""
    return im[np.clip(ys.astype(np.int64), 0, im.shape[0]-1), np.clip(xs.astype(np.int64), 0, im.shape[1]-1),...]

def _sample_nearest(im, xs, ys):
    return _sample_clipped(im, np.round(xs), np.round(ys))

def _sample_bilinear(im, xs, ys):
    # Convert arguments
    xs = np.asanyarray(xs)
    ys = np.asanyarray(ys)
    im = np.atleast_2d(np.asanyarray(im))

    if xs.shape != ys.shape:
        raise ValueError('Shape of xs and ys must match')
        
    # Split sample co-ords into floor and fractional part.
    floor_xs, floor_ys = np.floor(xs), np.floor(ys)
    frac_xs, frac_ys = xs - floor_xs, ys - floor_ys
    
    while len(im.shape) != len(frac_xs.shape):
        frac_xs = np.repeat(frac_xs[...,np.newaxis], im.shape[len(frac_xs.shape)], len(frac_xs.shape))
        frac_ys = np.repeat(frac_ys[...,np.newaxis], im.shape[len(frac_ys.shape)], len(frac_ys.shape))
    
    # Do x-wise sampling
    lower = (1.0 - frac_xs) * _sample_clipped(im, floor_xs, floor_ys) + frac_xs * _sample_clipped(im, floor_xs+1, floor_ys)
    upper = (1.0 - frac_xs) * _sample_clipped(im, floor_xs, floor_ys+1) + frac_xs * _sample_clipped(im, floor_xs+1, floor_ys+1)
    
    return ((1.0 - frac_ys) * lower + frac_ys * upper).astype(im.dtype)

def _sample_lanczos(im, xs, ys):
    # Convert arguments
    xs = np.asanyarray(xs)
    ys = np.asanyarray(ys)
    im = np.atleast_2d(np.asanyarray(im))

    if xs.shape != ys.shape:
        raise ValueError('Shape of xs and ys must match')
        
    # Split sample co-ords into floor part
    floor_xs, floor_ys = np.floor(xs), np.floor(ys)
    frac_xs, frac_ys = xs - floor_xs, ys - floor_ys

    a = 3.0

    def _l(x):
        # Note: NumPy's sinc function returns sin(pi*x) / (pi*x)
        return np.sinc(x) * np.sinc(x/a)

    S = None
    for dx in np.arange(-a+1, a+1):
        Lx = _l(frac_xs - dx)
        for dy in np.arange(-a+1, a+1):
            Ly = _l(frac_ys - dy)

            weight = Lx * Ly
            while len(im.shape) != len(weight.shape):
                weight = np.repeat(weight[...,np.newaxis], im.shape[len(weight.shape)], len(weight.shape))

            contrib = weight * _sample_clipped(im, floor_xs+dx, floor_ys+dy)
            if S is None:
                S = contrib
            else:
                S += contrib

    return S

def sample(im, xs, ys, method=DEFAULT_SAMPLE_METHOD):
    """Sample image at (x,y) given by elements of *xs* and *ys*. Both *xs* and *ys*
    must have identical shape and output will have this same shape. The
    location (x,y) refers to the *centre* of ``im[y,x]``. Samples at fractional
    locations are calculated using the method specified by *method*

    :param im: array to sample from
    :param xs: x co-ordinates to sample
    :param ys: y co-ordinates to sample
    :param method: one of 'bilinear', 'lanczos' or 'nearest'

    :raise ValueError: if ``xs`` and ``ys`` have differing shapes
    """
    if method == 'bilinear':
        return _sample_bilinear(im, xs, ys)
    elif method == 'lanczos':
        return _sample_lanczos(im, xs, ys)
    elif method == 'nearest':
        return _sample_nearest(im, xs, ys)
    
    raise NotImplementedError('Sampling method "{0}" is not implemented.'.format(method))

def scale(im, shape, method=DEFAULT_SAMPLE_METHOD):
    """Return a resampled version of *im* scaled to *shape*.

    Since the centre of pixel (x,y) has co-ordinate (x,y) the extent of *im* is
    actually :math:`x \in (-0.5, w-0.5]` and :math:`y \in (-0.5, h-0.5]`
    where (y,x) is ``im.shape``. This returns a sampled version of *im* that
    has the same extent as a *shape*-sized array.

    """
    # Original width and height (including half pixel)
    sh, sw = im.shape[:2]

    # New width and height (including half pixel)
    dh, dw = shape[:2]

    # Mapping from destination pixel (dx, dy) to im pixel (sx,sy) is:
    #
    #   x(dx) = (dx+0.5)*sw/dw - 0.5
    #   y(dy) = (dy+0.5)*sh/dh - 0.5
    #
    # which is a linear scale and offset transformation. So, for example, to
    # check that the extent dx in (-0.5, dw-0.5] maps to sx in (-0.5, sw-0.5]:
    #
    #   x(-0.5)     = (-0.5+0.5)*sw/dw - 0.5 = -0.5
    #   x(dw-0.5)   = (dw-0.5+0.5)*sw/dw - 0.5 = sw - 0.5

    dxs, dys = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    xscale = float(sw) / float(dw)
    yscale = float(sh) / float(dh)

    sxs = xscale * (dxs + 0.5) - 0.5
    sys = yscale * (dys + 0.5) - 0.5

    return sample(im, sxs, sys, method)

def _phase_image(xs, ys, unwrap=True):
    slices = []
    for ddx, ddy in zip(DTHETA_DX_2D, DTHETA_DY_2D):
        slice_phase = ddx * xs + ddy * ys
        if unwrap:
            slices.append(np.exp(-1j * slice_phase))
        else:
            slices.append(np.exp(1j * slice_phase))
    return np.dstack(slices)

def sample_highpass(im, xs, ys, method=DEFAULT_SAMPLE_METHOD):
    """As :py:func:`sample` except that the highpass image is first phase
    shifted to be centred on approximately DC.

    """
    # phase unwrap
    X, Y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    im_unwrap = im * _phase_image(X, Y, True)
    
    # sample
    im_sampled = sample(im_unwrap, xs, ys, method)
    
    # re-wrap
    return _phase_image(xs, ys, False) * im_sampled

def scale_highpass(im, shape, method=DEFAULT_SAMPLE_METHOD):
    """As :py:func:`sample` except that the highpass image is first phase
    shifted to be centred on approximately DC.

    """
    # Original width and height (including half pixel)
    sh, sw = im.shape[:2]

    # New width and height (including half pixel)
    dh, dw = shape[:2]

    # Mapping from destination pixel (dx, dy) to im pixel (sx,sy) is:
    #
    #   x(dx) = (dx+0.5)*sw/dw - 0.5
    #   y(dy) = (dy+0.5)*sh/dh - 0.5
    #
    # which is a linear scale and offset transformation. So, for example, to
    # check that the extent dx in (-0.5, dw-0.5] maps to sx in (-0.5, sw-0.5]:
    #
    #   x(-0.5)     = (-0.5+0.5)*sw/dw - 0.5 = -0.5
    #   x(dw-0.5)   = (dw-0.5+0.5)*sw/dw - 0.5 = sw - 0.5

    dxs, dys = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    xscale = float(sw) / float(dw)
    yscale = float(sh) / float(dh)

    sxs = xscale * (dxs + 0.5) - 0.5
    sys = yscale * (dys + 0.5) - 0.5

    # phase unwrap
    X, Y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    im_unwrap = im * _phase_image(X, Y, True)
    
    # sample
    im_sampled = sample(im_unwrap, sxs, sys, method)
    
    # re-wrap
    return im_sampled * _phase_image(sxs, sys, False)

# vim:sw=4:sts=4:et
