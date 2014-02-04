import os
import numpy as np

def regframes(name):
    """Load the *name* registration dataset and return source and reference frame."""
    frames = np.load(os.path.join(os.path.dirname(__file__), name + '.npz'))
    return frames['f1'], frames['f2']

def lena():
    """Return Lena in all her glory."""
    return np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']
