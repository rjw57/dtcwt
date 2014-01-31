"""
Convenience functions for plotting DTCWT-related objects.

The usage examples for functions in this module assume the following boilerplate:

.. ipython::

    In [0]: from pylab import *

    In [0]: import datasets

    In [0]: import dtcwt.plotting as dtcwtplt

    In [1]: import dtcwt.backend.backend_numpy as backend

    In [0]: transform2d = backend.Transform2d()

"""

import numpy as np
from matplotlib.pyplot import *

__all__ = (
    'overlay_quiver_DTCWT',
)

def overlay_quiver_DTCWT(image, vectorField, level, offset):
    """Overlays nicely coloured quiver plot of complex coefficients over original full-size image,
    providing a useful phase visualisation. vectorField is a single [MxNx6] numpy array of DTCWT 
    coefficients, level specifies the transform level of vectorField. Offset for DTCWT coefficients
    is typically 0.5. Should also work with other types of complex arrays (e.g., SLP coefficients),
    as long as the format is the same.

    Usage example:

    .. ipython::

        In [0]: lena = datasets.lena()

        In [0]: lena_t = transform2d.forward(lena, nlevels=3)

        In [0]: figure()

        @savefig gen-overlay_quiver_DTCWT.png
        In [0]: dtcwtplt.overlay_quiver_DTCWT(lena, lena_t.subbands[-1], 3, 0.5)

    .. codeauthor:: R. Anderson, 2005 (MATLAB)
    .. codeauthor:: S. C. Forshaw, 2014 (Python)
    """

    # You may wish to uncomment the following so that imshow() uses the full range of greyscale values
    imshow(image, cmap=cm.gray, clim=(0,1))

    # Set up the grid for the quiver plot
    g1 = np.kron(np.arange(0, vectorField[:,:,0].shape[0]).T, np.ones((1,vectorField[:,:,0].shape[1])))
    g2 = np.kron(np.ones((vectorField[:,:,0].shape[0], 1)), np.arange(0, vectorField[:,:,0].shape[1]))

    # Choose a coloUrmap
    cmap = cm.jet
    scalefactor = vectorField[-1,-1,:] = np.max(np.max(np.max(np.max(np.abs(vectorField)))))

    for sb in range(0, vectorField.shape[2]):
        thiscolour = cmap(sb / float(vectorField.shape[2])) # Select colour for this subband
        hq = quiver(g2*(2**level) + offset*(2**level), g1*(2**level) + offset*(2**level), np.real(vectorField[:,:,sb]), \
        np.imag(vectorField[:,:,sb]), color=thiscolour, scale=scalefactor*2**level)
        quiverkey(hq, image.shape[1]+75, 50 + sb*50, 200, "subband " + np.str(sb), coordinates='data', color=thiscolour)
        hold(True)

    return hq

