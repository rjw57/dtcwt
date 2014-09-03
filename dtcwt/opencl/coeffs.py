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
    low, _, high, _ = dtcwt.coeffs.biort(name)
    assert low.shape[0] % 2 == 1
    assert high.shape[0] % 2 == 1

    kernel_coeffs = np.zeros((max(low.shape[0], high.shape[0]),), cla.vec.float2)

    low_offset = (kernel_coeffs.shape[0] - low.shape[0])>>1
    kernel_coeffs['x'][low_offset:low_offset+low.shape[0]] = low.flatten()
    high_offset = (kernel_coeffs.shape[0] - high.shape[0])>>1
    kernel_coeffs['y'][high_offset:high_offset+high.shape[0]] = high.flatten()

    return kernel_coeffs

