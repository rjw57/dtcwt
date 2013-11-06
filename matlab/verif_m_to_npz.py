#!/usr/bin/env python

from scipy.io import loadmat
from numpy import savez

verif = loadmat('verification.mat')
verif = dict((k,v) for k, v in verif.iteritems() if not k.startswith('_'))

for idx, v in enumerate(verif['lena_Yh']):
    verif['lena_Yh_{0}'.format(idx)] = v[0]
del verif['lena_Yh']

for idx, v in enumerate(verif['lena_Yscale']):
    verif['lena_Yscale_{0}'.format(idx)] = v[0]
del verif['lena_Yscale']

#savez('../tests/verification.npz', **verif)

verifb = loadmat('verificationb.mat')
verifb = dict((k,v) for k, v in verifb.iteritems() if not k.startswith('_'))

for idx, v in enumerate(verifb['lena_Yhb']):
    verifb['lena_Yhb_{0}'.format(idx)] = v[0]
del verifb['lena_Yhb']

for idx, v in enumerate(verifb['lena_Yscaleb']):
    verifb['lena_Yscaleb_{0}'.format(idx)] = v[0]
del verifb['lena_Yscaleb']

allverif = dict(verif.items() + verifb.items())

savez('../tests/verification.npz', **allverif)
