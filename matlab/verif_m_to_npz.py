#!/usr/bin/env python

from scipy.io import loadmat
from numpy import savez

verif = loadmat('verification.mat')
savez('../tests/verification.npz', **verif)

