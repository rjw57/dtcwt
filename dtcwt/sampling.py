"""Rescaling and re-sampling high- and low-pass subbands.

"""

__all__ = (
    'sample', 'sample_highpass',
    'rescale', 'rescale_highpass',
    'upsample', 'upsample_highpass',
)

from dtcwt.lowlevel import reflect, asfarray

import numpy as np

_W0 = -3*np.pi/2.0
_W1 = -np.pi/2.0

#: The expected phase advances in the x-direction for each subband of the 2D transform
DTHETA_DX_2D = np.array((_W1, _W0, _W0, _W0, _W0, _W1))

#: The expected phase advances in the y-direction for each subband of the 2D transform
DTHETA_DY_2D = np.array((_W0, _W0, _W1, -_W1, -_W0, -_W0))

def _sample_clipped(im, xs, ys):
    """Truncated and symmetric sampling."""
    sym_xs = reflect(xs, -0.5, im.shape[1]-0.5).astype(np.int)
    sym_ys = reflect(ys, -0.5, im.shape[0]-0.5).astype(np.int)
    return im[sym_ys, sym_xs, ...]

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

def sample(im, xs, ys, method=None):
    """Sample image at (x,y) given by elements of *xs* and *ys*. Both *xs* and *ys*
    must have identical shape and output will have this same shape. The
    location (x,y) refers to the *centre* of ``im[y,x]``. Samples at fractional
    locations are calculated using the method specified by *method* (or
    ``'lanczos'`` if *method* is ``None``.)

    :param im: array to sample from
    :param xs: x co-ordinates to sample
    :param ys: y co-ordinates to sample
    :param method: one of 'bilinear', 'lanczos' or 'nearest'

    :raise ValueError: if ``xs`` and ``ys`` have differing shapes
    """
    if method is None:
        method = 'lanczos'

    if method == 'bilinear':
        return _sample_bilinear(im, xs, ys)
    elif method == 'lanczos':
        return _sample_lanczos(im, xs, ys)
    elif method == 'nearest':
        return _sample_nearest(im, xs, ys)
    
    raise NotImplementedError('Sampling method "{0}" is not implemented.'.format(method))

def rescale(im, shape, method=None):
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

def sample_highpass(im, xs, ys, method=None):
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

def rescale_highpass(im, shape, method=None):
    """As :py:func:`rescale` except that the highpass image is first phase
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

def _upsample_columns(X, method=None):
    """
    The centre of columns of X, an M-columned matrix, are assumed to have co-ordinates
    { 0, 1, 2, ... , M-1 } which means that the up-sampled matrix's columns should sample
    from { -0.25, 0.25, 0.75, ... , M-1.25 }. We can view that as an interleaved set of teo
    *convolutions* of X. The first, A, using a kernel equivalent to sampling the { -0.25, 0.75,
    1.75, 2.75, ... M-1.25 } columns and the second, B, sampling the { 0.25, 1.25, ... , M-0.75 }
    columns.
    """
    if method is None:
        method = 'lanczos'
    
    X = np.atleast_2d(asfarray(X))
    
    out_shape = list(X.shape)
    out_shape[1] *= 2
    output = np.zeros(out_shape, dtype=X.dtype)
    
    # Centres of sampling for A and B convolutions
    M = X.shape[1]
    A_columns = np.linspace(-0.25, M-1.25, M)
    B_columns = A_columns + 0.5
    
    # For A columns sample at x = ceil(x) - 0.25 with ceil(x) = { 0, 1, 2, ..., M-1 }
    # For B columns sample at x = floor(x) + 0.25 with floor(x) = { 0, 1, 2, ..., M-1 }
    int_columns = np.linspace(0, M-1, M)
    
    if method == 'lanczos':
        # Lanczos kernel width
        a = 3.0
        sample_offsets = np.arange(-a, a+1)
       
        # For A: if i = ceil(x) + di, => ceil(x) - i = -0.25 - di
        # For B: if i = floor(x) + di, => floor(x) - i = 0.25 - di
        l_as = np.sinc(-0.25-sample_offsets)*np.sinc((-0.25-sample_offsets)/a)   
        l_bs = np.sinc(0.25-sample_offsets)*np.sinc((0.25-sample_offsets)/a)
    elif method == 'nearest':
        # Nearest neighbour kernel width is 1
        sample_offsets = [0,]
        l_as = l_bs = [1,]
    elif method == 'bilinear':
        # Bilinear kernel width is technically 2 but we need to offset the kernels differently
        # for A and B columns:
        sample_offsets = [-1,0,1]
        l_as = [0.25, 0.75, 0]
        l_bs = [0, 0.75, 0.25]
    else:
        raise ValueError('Unknown interpolation mode: {0}'.format(mode))
    
    # Convolve
    for di, l_a, l_b in zip(sample_offsets, l_as, l_bs):
        columns = reflect(int_columns + di, -0.5, M-0.5).astype(np.int)
        
        output[:,0::2,...] += l_a * X[:,columns,...]
        output[:,1::2,...] += l_b * X[:,columns,...]
    
    return output
    
def upsample(image, method=None):
    """Specialised function to upsample an image by a factor of two using
    a specified sampling method. If *image* is an array of shape (NxMx...) then
    the output will have shape (2Nx2Mx...). Only rows and columns are
    upsampled, depth axes and greater are interpolated but are not upsampled.

    :param image: an array containing the image to upsample
    :param method: if non-None, a string specifying the sampling method to use.

    If *method* is ``None``, the default sampling method ``'lanczos'`` is used.
    The following sampling methods are supported:

    =========== ===========
    Name        Description 
    =========== ===========
    nearest     Nearest-neighbour sampling
    bilinear    Bilinear sampling
    lanczos     Lanczos sampling with window radius of 3
    =========== ===========
    """
    image = np.atleast_2d(asfarray(image))

    # The default '.T' operator doesn't quite do what we want since it
    # reverses the axes rather than only swapping the first two
    def _t(X):
        axes = np.arange(len(X.shape))
        axes[:2] = (1,0)
        return np.transpose(X, axes)

    return _upsample_columns(_t(_upsample_columns(_t(image), method)), method)

def upsample_highpass(im, method=None):
    """As :py:func:`upsample` except that the highpass image is first phase
    rolled so that the filter has approximate DC centre frequency. The upshot
    is that this is the function to use when re-sampling complex subband
    images.

    """
    im = np.atleast_2d(asfarray(im))

    # Sampled co-ordinates
    dxs, dys = np.meshgrid(np.arange(im.shape[1]*2), np.arange(im.shape[0]*2))
    sxs = 0.5 * (dxs + 0.5) - 0.5
    sys = 0.5 * (dys + 0.5) - 0.5

    # phase unwrap
    X, Y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    im_unwrap = im * _phase_image(X, Y, True)
    
    # sample
    im_sampled = upsample(im_unwrap, method)
    
    # re-wrap
    return im_sampled * _phase_image(sxs, sys, False)

# vim:sw=4:sts=4:et
