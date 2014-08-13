import os
import numpy as np
import PIL.Image as Image

def regframes(name):
    """Load the *name* registration dataset and return source and reference frame."""
    frames = np.load(os.path.join(os.path.dirname(__file__), name + '.npz'))
    return frames['f1'], frames['f2']

def lena():
    """Return Lena in all her glory."""
    return np.load(os.path.join(os.path.dirname(__file__), 'lena.npz'))['lena']

def traffic_hd():
    """Return a 1080p image of some traffic."""
    return np.asarray(Image.open(os.path.join(os.path.dirname(__file__), 'traffic_hd.jpg')))

