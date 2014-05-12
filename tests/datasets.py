import os
import sys
import numpy as np

# HACK: PyPy Numpy requires a bit of monkey patching.
# See https://bugs.pypy.org/issue1766
if 'PyPy' in sys.version and np.frombuffer is None:
    np.frombuffer = np.fromstring

def regframes(name):
    """Load the *name* registration dataset and return source and reference frame."""
    frames = np.load(os.path.join(os.path.dirname(__file__), name + '.npz'))
    return frames['f1'], frames['f2']

def lena():
    """Return Lena in all her glory."""
    return np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']
