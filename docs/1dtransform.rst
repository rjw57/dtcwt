1D transform
------------

This example generates two 1D random walks and demonstrates reconstructing them
using the forward and inverse 1D transforms. Note that
:py:func`dtcwt.Transform1d.forward` and :py:func:`dtcwt.Transform1d.inverse`
will transform columns of an input array independently

.. plot::
    :include-source: true

    from matplotlib.pylab import *
    import dtcwt

    # Generate a 300x2 array of a random walk
    vecs = np.cumsum(np.random.rand(300,2) - 0.5, 0)

    # Show input
    figure()
    plot(vecs)
    title('Input')

    # 1D transform, 5 levels
    transform = dtcwt.Transform1d()
    vecs_t = transform.forward(vecs, nlevels=5)

    # Show level 2 highpass coefficient magnitudes
    figure()
    plot(np.abs(vecs_t.highpasses[1]))
    title('Level 2 wavelet coefficient magnitudes')

    # Show last level lowpass image
    figure()
    plot(vecs_t.lowpass)
    title('Lowpass signals')

    # Inverse
    vecs_recon = transform.inverse(vecs_t)

    # Show output
    figure()
    plot(vecs_recon)
    title('Output')

    # Show error
    figure()
    plot(vecs_recon - vecs)
    title('Reconstruction error')

    print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs - vecs_recon))))


