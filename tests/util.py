import numpy as np

TOLERANCE = 1e-12

def assert_almost_equal(a, b, tolerance=TOLERANCE):
    md = np.abs(a-b).max()
    if md <= tolerance:
        return

    raise AssertionError(
            'Arrays differ by a maximum of {0} which is greater than the tolerance of {1}'.
            format(md, tolerance))

def summarise_mat(M, apron=8):
    """HACK to provide a 'summary' matrix consisting of the corners of the
    matrix and summed versions of the sub matrices.
    
    N.B. Keep this in sync with matlab/verif_m_to_npz.py.

    """
    centre = M[apron:-apron,apron:-apron,...]
    centre_sum = np.mean(np.mean(centre, axis=0, keepdims=True), axis=1, keepdims=True)

    return np.vstack((
        np.hstack((M[:apron,:apron,...], np.mean(M[:apron,apron:-apron,...], axis=1, keepdims=True), M[:apron,-apron:,...])),
        np.hstack((np.mean(M[apron:-apron,:apron,...], axis=0, keepdims=True), centre_sum, np.mean(M[apron:-apron,-apron:,...], axis=0, keepdims=True))),
        np.hstack((M[-apron:,:apron,...], np.mean(M[-apron:,apron:-apron,...], axis=1, keepdims=True), M[-apron:,-apron:,...])),
    ))
