""" Useful utilities for testing the 2-D DTCWT with synthetic images"""

import functools
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

def asfarray(X):
    """Similar to :py:func:`numpy.asfarray` except that this function tries to
    preserve the original datatype of X if it is already a floating point type
    and will pass floating point arrays through directly without copying.

    """
    X = np.asanyarray(X)
    return np.asfarray(X, dtype=X.dtype)

def appropriate_complex_type_for(X):
    """Return an appropriate complex data type depending on the type of X. If X
    is already complex, return that, if it is floating point return a complex
    type of the appropriate size and if it is integer, choose an complex
    floating point type depending on the result of :py:func:`numpy.asfarray`.

    """
    X = asfarray(X)
    
    if np.issubsctype(X.dtype, np.complex64) or np.issubsctype(X.dtype, np.complex128):
        return X.dtype
    elif np.issubsctype(X.dtype, np.float32):
        return np.complex64
    elif np.issubsctype(X.dtype, np.float64):
        return np.complex128

    # God knows, err on the side of caution
    return np.complex128

def as_column_vector(v):
    """Return *v* as a column vector with shape (N,1).
    
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v

def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx* and
    *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers + 0.5), the
    ramps will have repeated max and min samples.
   
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    
    """

    # Copy x to avoid in-place modification
    y = np.array(x, copy=True)

    # Reflect y in maxx.
    t = y > maxx
    y[t] = (2*maxx - y[t]).astype(y.dtype)

    while np.any(y < minx):
        # Reflect y in minx.
        t = y < minx
        y[t] = (2*minx - y[t]).astype(y.dtype)

        # Reflect y in maxx.
        t = y > maxx
        y[t] = (2*maxx - y[t]).astype(y.dtype)

    return y

# note that this decorator ignores **kwargs
# From https://wiki.python.org/moin/PythonDecoratorLibrary#Alternate_memoize_as_nested_functions
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer
