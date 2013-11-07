#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat
from numpy import savez

def summarise_mat(M, apron=8):
    """HACK to provide a 'summary' matrix consisting of the corners of the
    matrix and summed versions of the sub matrices."""

    centre = M[apron:-apron,apron:-apron,...]
    centre_sum = np.mean(np.mean(centre, axis=0, keepdims=True), axis=1, keepdims=True)

    return np.vstack((
        np.hstack((M[:apron,:apron,...], np.mean(M[:apron,apron:-apron,...], axis=1, keepdims=True), M[:apron,-apron:,...])),
        np.hstack((np.mean(M[apron:-apron,:apron,...], axis=0, keepdims=True), centre_sum, np.mean(M[apron:-apron,-apron:,...], axis=0, keepdims=True))),
        np.hstack((M[-apron:,:apron,...], np.mean(M[-apron:,apron:-apron,...], axis=1, keepdims=True), M[-apron:,-apron:,...])),
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
