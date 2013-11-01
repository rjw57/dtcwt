""" Useful utilities for testing the 2-D DTCWT with synthetic images"""

__all__ = ( 'drawedge', 'drawcirc',)

import numpy as np


def drawedge(theta,r,w,N):
    """Generate an image of size N * N pels, of an edge going from 0 to 1
    in height at theta degrees to the horizontal (top of image = 1 if angle = 0).
    r is a two-element vector, it is a coordinate in ij coords through
    which the step should pass.
    The shape of the intensity step is half a raised cosine w pels wide (w>=1).
    
    T. E . Gale's enhancement to drawedge() for MATLAB, transliterated 
    to Python by S. C. Forshaw, Nov. 2013. """
    
    # convert theta from degrees to radians
    thetar = np.array(theta * np.pi / 180)  
    
    # Calculate image centre from given width
    imCentre = (np.array([N,N]).T - 1) / 2 + 1 
    
    # Calculate values to subtract from the plane
    r = np.array([np.cos(thetar), np.sin(thetar)])*(-1) * (r - imCentre) 
    print(r)
    # check width of raised cosine section
    w = np.maximum(1,w)
    
    
    ramp = np.arange(0,N) - (N+1)/2
    hgrad = np.sin(thetar)*(-1) * np.ones([N,1])
    vgrad = np.cos(thetar)*(-1) * np.ones([1,N])
    plane = ((hgrad * ramp) - r[0]) + ((ramp * vgrad).T - r[1])
    x = 0.5 + 0.5 * np.sin(np.minimum(np.maximum(plane*(np.pi/w), np.pi/(-2)), np.pi/2))
    
    return x

def drawcirc(r,w,du,dv,N):
    
    """Generate an image of size N*N pels, containing a circle 
    radius r pels and centred at du,dv relative
    to the centre of the image.  The edge of the circle is a cosine shaped 
    edge of width w (from 10 to 90% points).
    
    Python implementation by S. C. Forshaw, November 2013."""
    
    # check value of w to avoid dividing by zero
    w = np.maximum(w,1)
    
    #x plane
    x = np.ones([N,1]) * ((np.arange(0,N,1, dtype='float') - (N+1) / 2 - dv) / r)
    
    # y vector
    y = (((np.arange(0,N,1, dtype='float') - (N+1) / 2 - du) / r) * np.ones([1,N])).T

    # Final circle image plane
    p = 0.5 + 0.5 * np.sin(np.minimum(np.maximum((np.exp(np.array([-0.5]) * (x**2 + y**2)).T - np.exp((-0.5))) * (r * 3 / w), np.pi/(-2)), np.pi/2))
    return p
