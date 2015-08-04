#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat
from numpy import savez_compressed

def _mean(a, axis=None, *args, **kwargs):
    """Equivalent to numpy.mean except that the axis along which the mean is taken is not removed."""

    rv = np.mean(a, axis=axis, *args, **kwargs)

    if axis is not None:
        rv = np.expand_dims(rv, axis)

    return rv

def centre_indices(ndim=2,apron=8):
    """Returns the centre indices for the correct number of dimension
    """
    return tuple([slice(apron,-apron) for i in range(ndim)])

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

def summarise_cube(M, apron=4):
    """Provide a summary cube, extending  summarise_mat to 3D
    """
    return np.dstack(
        [summarise_mat(M[:,:,i,...], apron) for i in range(M.shape[-2])]
    )

verif_temp = loadmat('verification.mat')
verif = dict((k,v) for k, v in verif_temp.items() if (not k.startswith('_') and not k.startswith('qbgn')))
verif_cube = dict((k,v) for k, v in verif_temp.items() if (not k.startswith('_') and k.startswith('qbgn')))
del verif_temp

for idx, v in enumerate(verif['mandrill_Yh']):
    verif['mandrill_Yh_{0}'.format(idx)] = v[0]
del verif['mandrill_Yh']

for idx, v in enumerate(verif['mandrill_Yscale']):
    verif['mandrill_Yscale_{0}'.format(idx)] = v[0]
del verif['mandrill_Yscale']

for idx, v in enumerate(verif['mandrill_Yhb']):
    verif['mandrill_Yhb_{0}'.format(idx)] = v[0]
del verif['mandrill_Yhb']

for idx, v in enumerate(verif['mandrill_Yscaleb']):
    verif['mandrill_Yscaleb_{0}'.format(idx)] = v[0]
del verif['mandrill_Yscaleb']

for idx, v in enumerate(verif_cube['qbgn_Yh']):
    verif_cube['qbgn_Yh_{0}'.format(idx)] = v[0]
del verif_cube['qbgn_Yh']

for idx, v in enumerate(verif_cube['qbgn_Yscale']):
    verif_cube['qbgn_Yscale_{0}'.format(idx)] = v[0]
del verif_cube['qbgn_Yscale']

summaries = dict((k, summarise_mat(v)) for k, v in verif.items())
for k,v in verif_cube.items():
    summaries[k] = summarise_cube(v)

savez_compressed('../tests/verification.npz', **summaries)

# Convert qbgn.mat -> qbgn.npz
savez_compressed('../tests/qbgn.npz', **loadmat('qbgn.mat'))
