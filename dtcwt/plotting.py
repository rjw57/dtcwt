"""
Convenience functions for plotting DTCWT-related objects.

"""

from __future__ import absolute_import

import numpy as np
from matplotlib.pyplot import *

__all__ = (
    'overlay_quiver_DTCWT',
)

def overlay_quiver_DTCWT(image, vectorField, level, offset):
    """Overlays nicely coloured quiver plot of complex coefficients over original full-size image,
    providing a useful phase visualisation.

    :param image: array holding grayscale values on the interval [0, 255] to display
    :param vectorField: a single [MxNx6] numpy array of DTCWT coefficients
    :param level: the transform level (1-indexed) of *vectorField*.
    :param offset: Offset for DTCWT coefficients (typically 0.5)

    .. note::

        The *level* parameter is 1-indexed meaning that the third level has
        index "3". This is unusual in Python but is kept for compatibility
        with similar MATLAB routines.

    Should also work with other types of complex arrays (e.g., SLP
    coefficients), as long as the format is the same.

    Usage example:

    .. plot::
        :include-source: true

        import dtcwt
        import dtcwt.plotting as plotting

        lena = datasets.lena()

        transform2d = dtcwt.Transform2d()
        lena_t = transform2d.forward(lena, nlevels=5)

        plotting.overlay_quiver_DTCWT(lena*255, lena_t.highpasses[-1], 5, 0.5)

    .. codeauthor:: R. Anderson, 2005 (MATLAB)
    .. codeauthor:: S. C. Forshaw, 2014 (Python)
    """

    # Make sure imshow() uses the full range of greyscale values
    imshow(image, cmap=cm.gray, clim=(0,255))

    # Set up the grid for the quiver plot
    g1 = np.kron(np.arange(0, vectorField[:,:,0].shape[0]).T, np.ones((1,vectorField[:,:,0].shape[1])))
    g2 = np.kron(np.ones((vectorField[:,:,0].shape[0], 1)), np.arange(0, vectorField[:,:,0].shape[1]))

    # Choose a coloUrmap
    cmap = cm.spectral
    scalefactor = np.max(np.max(np.max(np.max(np.abs(vectorField)))))
    vectorField[-1,-1,:] = scalefactor

    for sb in range(0, vectorField.shape[2]):
        thiscolour = cmap(sb / float(vectorField.shape[2])) # Select colour for this subband
        hq = quiver(g2*(2**level) + offset*(2**level), g1*(2**level) + offset*(2**level), np.real(vectorField[:,:,sb]), \
        np.imag(vectorField[:,:,sb]), color=thiscolour, scale=scalefactor*2**level)
        quiverkey(hq, 1.05, 0.9-0.05*sb, 0, "subband " + np.str(sb), coordinates='axes', color=thiscolour, labelcolor=thiscolour, labelpos='E')

    return hq
