#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat
from numpy import savez

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

verif = loadmat('verification.mat')
verif = dict((k,v) for k, v in verif.iteritems() if not k.startswith('_'))

for idx, v in enumerate(verif['lena_Yh']):
    verif['lena_Yh_{0}'.format(idx)] = v[0]
del verif['lena_Yh']

for idx, v in enumerate(verif['lena_Yscale']):
    verif['lena_Yscale_{0}'.format(idx)] = v[0]
del verif['lena_Yscale']

for idx, v in enumerate(verif['lena_Yhb']):
    verif['lena_Yhb_{0}'.format(idx)] = v[0]
del verif['lena_Yhb']

for idx, v in enumerate(verif['lena_Yscaleb']):
    verif['lena_Yscaleb_{0}'.format(idx)] = v[0]
del verif['lena_Yscaleb']

summaries = dict((k, summarise_mat(v)) for k, v in verif.iteritems())

savez('../tests/verification.npz', **summaries)
