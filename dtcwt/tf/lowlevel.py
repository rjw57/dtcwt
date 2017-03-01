from __future__ import absolute_import

import tensorflow as tf
import numpy as np

def colfilter(X, h):
    """Filter the columns of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [1 0].
    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    m = h.get_shape().as_list()[0]
    m2 = m // 2

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns)
    X = tf.pad(X, [[0, 0], [m2, m2], [0, 0]], 'SYMMETRIC')

    # Reshape h to be a col filter. We have to flip h too as the tf conv2d
    # operation is cross-correlation, not true convolution
    h = tf.reshape(h[::-1], [-1, 1, 1, 1])

    # Reshape X from [batch, rows, cols] to [batch, rows, cols, 1] for conv2d
    X = tf.expand_dims(X, axis=-1)

    Y = tf.nn.conv2d(X, h, strides=[1, 1, 1, 1], padding='VALID')

    # Drop the last dimension
    return tf.unstack(Y, num=1, axis=-1)[0]

def rowfilter(X, h):
    """Filter the rows of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [0 1].
    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    m = h.get_shape().as_list()[0]
    m2 = m // 2

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns)
    X = tf.pad(X, [[0, 0], [0, 0], [m2, m2]], 'SYMMETRIC')

    # Reshape h to be a row filter. We have to flip h too as the tf conv2d
    # operation is cross-correlation, not true convolution
    h = tf.reshape(h[::-1], [1, -1, 1, 1])

    # Reshape X from [batch, rows, cols] to [batch, rows, cols, 1] for conv2d
    X = tf.expand_dims(X, axis=-1)

    Y = tf.nn.conv2d(X, h, strides=[1, 1, 1, 1], padding='VALID')

    # Drop the last dimension
    return tf.unstack(Y, num=1, axis=-1)[0]


def coldfilt(X, ha, hb, a_first=True):
    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha). 
    ha operates on the odd samples of X and hb on the even samples.  
    Both filters should be even length, and h should be approx linear
    phase with a quarter sample (i.e. an :math:`e^{j \pi/4}`) advance from 
    its mid pt (i.e. :math:`|h(m/2)| > |h(m/2 + 1)|`).
    .. code-block:: text
                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a
    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  
    Symmetric extension with repeated end samples is used on the composite X columns
    before each filter is applied.
    Raises ValueError if the number of rows in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    r, c = X.get_shape().as_list()[1:]
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    m = ha.get_shape().as_list()[0]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns).
    X = tf.pad(X, [[0, 0], [m, m], [0, 0]], 'SYMMETRIC')    

    # Take the odd and even columns of X
    X_odd = tf.expand_dims(X[:,2:r+2*m-2:2,:], axis=-1)
    X_even =tf.expand_dims(X[:,3:r+2*m-2:2,:], axis=-1)

    # Transform ha and hb to be col filters. We must reverse them as tf conv is
    # cross correlation, not true convolution
    ha = tf.reshape(ha[::-1], [m,1,1,1])
    hb = tf.reshape(hb[::-1], [m,1,1,1])

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    a_rows = tf.nn.conv2d(X_odd, ha, strides=[1,2,1,1], padding='VALID')
    b_rows = tf.nn.conv2d(X_even, hb, strides=[1,2,1,1], padding='VALID')
    
    # We interleave the two results into a tensor of size [Batch, r/2, c]
    # Concat a_rows and b_rows (both of shape [Batch, r/4, c, 1])     
    Y = tf.cond(tf.reduce_sum(ha*hb) > 0,
                lambda: tf.concat([a_rows,b_rows],axis=-1),
                lambda: tf.concat([b_rows,a_rows],axis=-1))
        
    # Permute result to be shape [Batch, r/4, 2, c]
    Y = tf.transpose(Y, perm=[0,1,3,2])
    
    # Reshape result to be shape [Batch, r/2, c]. This reshaping interleaves
    # the columns
    Y = tf.reshape(Y, [-1, r2, c])   
    
    return Y


def rowdfilt(X, ha, hb):
    """Filter the rows of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`).
    .. code-block:: text
                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a
    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  Symmetric
    extension with repeated end samples is used on the composite X rows
    before each filter is applied.
    Raises ValueError if the number of columns in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.
    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    r, c = X.get_shape().as_list()[1:]
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    m = ha.get_shape().as_list()[0]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the rows).
    # SYMMETRIC extension means the edge sample is repeated twice, whereas
    # REFLECT only has the edge sample once    
    X = tf.pad(X, [[0, 0], [0, 0], [m, m]], 'SYMMETRIC')

    # Take the odd and even columns of X
    X_odd = tf.expand_dims(X[:,:,2:c+2*m-2:2], axis=-1)
    X_even =tf.expand_dims(X[:,:,3:c+2*m-2:2], axis=-1)

    # Transform ha and hb to be col filters. We must reverse them as tf conv is
    # cross correlation, not true convolution
    ha = tf.reshape(ha[::-1], [m,1,1,1])
    hb = tf.reshape(hb[::-1], [m,1,1,1])

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    a_cols = tf.nn.conv2d(X_odd, ha, strides=[1,1,2,1], padding='VALID')
    b_cols = tf.nn.conv2d(X_even, hb, strides=[1,1,2,1], padding='VALID')
    
    # We interleave the two results into a tensor of size [Batch, r/2, c]
    # Concat a_cols and b_cols (both of shape [Batch, r, c/4, 1])      
    Y = tf.cond(tf.reduce_sum(ha*hb) > 0,
                lambda: tf.concat([a_cols,b_cols],axis=-1),
                lambda: tf.concat([b_cols,a_cols],axis=-1))
        
    # Reshape result to be shape [Batch, r, c/2]. This reshaping interleaves
    # the columns
    Y = tf.reshape(Y, [-1, r, c2])   
    
    return Y

