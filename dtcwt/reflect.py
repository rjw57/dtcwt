import numpy as np

def reflect(x, minx, maxx):
    """Reflect the values in matrix x about the scalar values minx and maxx.
    Hence a vector x containing a long linearly increasing series is converted
    into a waveform which ramps linearly up and down between minx and maxx.  If
    x contains integers and minx and maxx are (integers + 0.5), the ramps will
    have repeated max and min samples.
   
    Nick Kingsbury, Cambridge University, January 1999.
    
    """

    # Copy x to avoid in-place modification
    y = np.array(x, copy=True)

    # Reflect y in maxx.
    y[y > maxx] = 2*maxx - y[y > maxx]

    while np.any(y < minx):
        # Reflect y in minx.
        y[y < minx] = 2*minx - y[y < minx]

        # Reflect y in maxx.
        y[y > maxx] = 2*maxx - y[y > maxx]

    return y

# vim:sw=4:sts=4:et
