import functools
import numpy as np

from nose import SkipTest
from dtcwt.backend.backend_opencl.lowlevel import NoCLPresentError

TOLERANCE = 1e-6

def assert_almost_equal(a, b, tolerance=TOLERANCE):
    md = np.abs(a-b).max()
    if md <= tolerance:
        return

    raise AssertionError(
            'Arrays differ by a maximum of {0} which is greater than the tolerance of {1}'.
            format(md, tolerance))

def _mean(a, axis=None, *args, **kwargs):
    """Equivalent to numpy.mean except that the axis along which the mean is taken is not removed."""

    rv = np.mean(a, axis=axis, *args, **kwargs)

    if axis is not None:
        rv = np.expand_dims(rv, axis)

    return rv

def summarise_mat(M, apron=8):
    """HACK to provide a 'summary' matrix consisting of the corners of the
    matrix and summed versions of the sub matrices.
    
    N.B. Keep this in sync with matlab/verif_m_to_npz.py.

    """
    centre = M[apron:-apron,apron:-apron,...]
    centre_sum = _mean(_mean(centre, axis=0), axis=1)

    return np.vstack((
        np.hstack((M[:apron,:apron,...], _mean(M[:apron,apron:-apron,...], axis=1), M[:apron,-apron:,...])),
        np.hstack((_mean(M[apron:-apron,:apron,...], axis=0), centre_sum, _mean(M[apron:-apron,-apron:,...], axis=0))),
        np.hstack((M[-apron:,:apron,...], _mean(M[-apron:,apron:-apron,...], axis=1), M[-apron:,-apron:,...])),
    ))

def skip_if_no_cl(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except NoCLPresentError:
            raise SkipTest('Skipping due to no CL library being present')
    return wrapper
