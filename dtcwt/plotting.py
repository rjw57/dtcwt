"""
Convenience functions for plotting DTCWT-related objects.

"""

from __future__ import absolute_import

import numpy as np
from matplotlib.pyplot import *

__all__ = (
    'overlay_quiver',
)

def overlay_quiver(image, vectorField, level, offset):
    """Overlays nicely coloured quiver plot of complex coefficients over original full-size image,
    providing a useful phase visualisation. image should be of type UINT8. vectorField is a single 
    [MxNx6] numpy array of DTCWT coefficients, level specifies the transform level of vectorField. 
    Offset for DTCWT coefficients is typically 0.5. Should also work with other types of complex 
    arrays (e.g., SLP coefficients), as long as the format is the same.
    
    Usage example:
    
    .. ipython::
    
    In [0]: lena = datasets.lena()
    
    In [0]: lena_t = transform2d.forward(lena, nlevels=3)
    
    In [0]: figure()
    
    @savefig gen-overlay_quiver.png
    In [0]: dtcwtplt.overlay_quiver(lena, lena_t.subbands[-1], 3, 0.5)
    
    .. codeauthor:: R. Anderson, 2005 (MATLAB)
    .. codeauthor:: S. C. Forshaw, 2014 (Python)
    """

    # Make sure imshow() uses the full range of greyscale values
    imshow(image, cmap=cm.gray, clim=(0,255))
    hold(True)
       
    # Set up the grid for the quiver plot
    g1 = np.kron(np.arange(0, vectorField[:,:,0].shape[0]).T, np.ones((1,vectorField[:,:,0].shape[1])))
    g2 = np.kron(np.ones((vectorField[:,:,0].shape[0], 1)), np.arange(0, vectorField[:,:,0].shape[1]))

    # Choose a coloUrmap
    cmap = cm.spectral
    scalefactor = np.max(np.max(np.max(np.max(np.abs(vectorField)))))
    vectorField[-1,-1,:] = scalefactor
        
    print(scalefactor)
    for sb in range(0, vectorField.shape[2]):
        hold(True)
        thiscolour = cmap(sb / float(vectorField.shape[2])) # Select colour for this subband
        hq = quiver(g2*(2**level) + offset*(2**level), g1*(2**level) + offset*(2**level), np.real(vectorField[:,:,sb]), \
        np.imag(vectorField[:,:,sb]), color=thiscolour, scale=scalefactor*2**level)
        quiverkey(hq, 1.05, 1.00-0.035*sb, 0, "subband " + np.str(sb), coordinates='axes', color=thiscolour, labelcolor=thiscolour, labelpos='E')

    hold(False)
    return hq
