import theano.tensor as T

from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.numpy import Transform2d, Pyramid

def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False):
    t = Transform2d(biort=biort, qshift=qshift)
    r = t.forward(X, nlevels=nlevels, include_scale=include_scale)
    if include_scale:
        return r.lowpass, r.highpasses, r.scales
    else:
        return r.lowpass, r.highpasses
