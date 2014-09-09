"""Load wavelet coefficients in a form suitable for OpenCL convolution."""

import numpy as np
import dtcwt.coeffs

try:
    import pyopencl.array as cla
except ImportError:
    pass

def biort(name):
    """Return a host array of kernel coefficients for the named biorthogonal
    level 1 wavelet. The lowpass coefficients are loaded into the 'x' field of
    the returned array and the highpass are loaded into the 'y' field. The
    returned array is suitable for passing to Convolution1D().
    """
    low, _, high = dtcwt.coeffs.biort(name)[:3]
    assert low.shape[0] % 2 == 1
    assert high.shape[0] % 2 == 1

    kernel_coeffs = np.zeros((max(low.shape[0], high.shape[0]),), cla.vec.float2)

    low_offset = (kernel_coeffs.shape[0] - low.shape[0])>>1
    kernel_coeffs['x'][low_offset:low_offset+low.shape[0]] = low.flatten()
    high_offset = (kernel_coeffs.shape[0] - high.shape[0])>>1
    kernel_coeffs['y'][high_offset:high_offset+high.shape[0]] = high.flatten()

    return kernel_coeffs

def qshift(name):
    """Return a pair host array of kernel coefficients for the named
    quarter-sample shift wavelet. The even and odd (1-indexed) h0 coefficients
    are loaded into the 'x' and 'y' fields of the first returned array and the
    even and odd (1-indexed) h1 coefficients are loaded into the 'x' and 'y'
    fields of the second.  The returned arrays are suitable for passing to
    Convolution1D().

    """
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = dtcwt.coeffs.qshift(name)[:8]
    low, high = h0b, h1b

    if np.sum(low*low[::-1]) <= 0:
        low = low[::-1]
    if np.sum(high*high[::-1]) <= 0:
        high = high[::-1]

    low_odd = low.flatten()[0::2]
    low_even = low.flatten()[1::2]
    high_odd = high.flatten()[0::2]
    high_even = high.flatten()[1::2]

    assert low_odd.shape[0] % 2 == 1
    assert high_odd.shape[0] % 2 == 1
    assert high_odd.shape[0] % 2 == 1
    assert low_odd.shape[0] % 2 == 1

    kernel_coeffs = np.zeros((max(low.shape[0]>>1, high.shape[0]>>1),), cla.vec.float4)

    low_odd_offset = (kernel_coeffs.shape[0] - low_odd.shape[0])>>1
    kernel_coeffs['x'][low_odd_offset:low_odd_offset+low_odd.shape[0]] = low_odd.flatten()
    low_even_offset = (kernel_coeffs.shape[0] - low_even.shape[0])>>1
    kernel_coeffs['y'][low_even_offset:low_even_offset+low_even.shape[0]] = low_even.flatten()
    high_odd_offset = (kernel_coeffs.shape[0] - high_odd.shape[0])>>1
    kernel_coeffs['z'][high_odd_offset:high_odd_offset+high_odd.shape[0]] = high_odd.flatten()
    high_even_offset = (kernel_coeffs.shape[0] - high_even.shape[0])>>1
    kernel_coeffs['w'][high_even_offset:high_even_offset+high_even.shape[0]] = high_even.flatten()

    return kernel_coeffs

