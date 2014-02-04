1D transform
------------

This example generates two 1D random walks and demonstrates reconstructing them
using the forward and inverse 1D transforms. Note that
:py:func:`dtcwt.dtwavexfm` and :py:func:`dtcwt.dtwaveifm` will transform
columns of an input array independently::

    import numpy as np
    from matplotlib.pyplot import *

    # Generate a 300x2 array of a random walk
    vecs = np.cumsum(np.random.rand(300,2) - 0.5, 0)

    # Show input
    figure(1)
    plot(vecs)
    title('Input')

    import dtcwt

    # 1D transform
    Yl, Yh = dtcwt.dtwavexfm(vecs)

    # Inverse
    vecs_recon = dtcwt.dtwaveifm(Yl, Yh)

    # Show output
    figure(2)
    plot(vecs_recon)
    title('Output')

    # Show error
    figure(3)
    plot(vecs_recon - vecs)
    title('Reconstruction error')

    print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs - vecs_recon))))

    show()


