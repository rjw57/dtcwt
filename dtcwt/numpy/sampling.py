import numpy as np

from dtcwt.utils import reflect

def _sample_clipped(im, xs, ys):
    """Truncated and symmetric sampling."""
    sym_xs = reflect(xs, -0.5, im.shape[1]-0.5).astype(np.int)
    sym_ys = reflect(ys, -0.5, im.shape[0]-0.5).astype(np.int)
    return im[sym_ys, sym_xs, ...]

def sample_nearest(im, xs, ys):
    return _sample_clipped(im, np.round(xs), np.round(ys))

def sample_bilinear(im, xs, ys):
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

def sample_lanczos(im, xs, ys):
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

