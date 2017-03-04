from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from dtcwt.utils import asfarray, as_column_vector

try:
    import tensorflow as tf
    _HAVE_TF = True
except ImportError:
    _HAVE_TF = False

def _as_row_tensor(h):
    if isinstance(h, tf.Tensor):
        h = tf.reshape(h, [1, -1])
    else:
        h = as_column_vector(h).T
        h = tf.constant(h, tf.float32)
    return h

def _as_col_tensor(h):
    if isinstance(h, tf.Tensor):
        h = tf.reshape(h, [-1, 1])
    else:
        h = as_column_vector(h)
        h = tf.constant(h, tf.float32)
    return h

def _conv_2d(X, h, strides=[1,1,1,1]):
    """Perform 2d convolution in tensorflow. X will to be manipulated to be of
    shape [batch, height, width, ch], and h to be of shape 
    [height, width, ch, num]. This function does the necessary reshaping before 
    calling the conv2d function, and does the reshaping on the output, returning 
    Y of shape [batch, height, width]"""
    
    # Check the shape of X is what we expect
    if len(X.shape) != 3:
        raise ValueError('X needs to be of shape [batch, height, width] for conv_2d') 
    
    # Check the shape of h is what we expect
    if len(h.shape) != 2:
        raise ValueError('Filter inputs must only have height and width for conv_2d')

    # Add in the unit dimensions for conv
    X = tf.expand_dims(X, axis=-1)
    h = tf.expand_dims(tf.expand_dims(h, axis=-1),axis=-1)

    # Have to reverse h as tensorflow 2d conv is actually cross-correlation
    h = tf.reverse(h, axis=[0,1])
    Y = tf.nn.conv2d(X, h, strides=strides, padding='VALID')

    # Remove the final dimension, returning a result of shape [batch, height, width]
    Y = tf.squeeze(Y, axis=-1)

    return Y

def _conv_2d_transpose(X, h, out_shape, strides=[1,1,1,1]):
    """Perform 2d convolution in tensorflow. X will to be manipulated to be of
    shape [batch, height, width, ch], and h to be of shape 
    [height, width, ch, num]. This function does the necessary reshaping before 
    calling the conv2d function, and does the reshaping on the output, returning 
    Y of shape [batch, height, width]"""
    
    # Check the shape of X is what we expect
    if len(X.shape) != 3:
        raise ValueError('X needs to be of shape [batch, height, width] for conv_2d') 
    # Check the shape of h is what we expect
    if len(h.shape) != 2:
        raise ValueError('Filter inputs must only have height and width for conv_2d')

    # Add in the unit dimensions for conv
    X = tf.expand_dims(X, axis=-1)
    h = tf.expand_dims(tf.expand_dims(h, axis=-1),axis=-1)

    # Have to reverse h as tensorflow 2d conv is actually cross-correlation
    h = tf.reverse(h, axis=[0,1])
    # Transpose h as we will be using the transpose convolution
    h = tf.transpose(h, perm=[1, 0, 2, 3])

    Y = tf.nn.conv2d(X, h, output_shape=out_shape, strides=strides, padding='VALID')

    # Remove the final dimension, returning a result of shape [batch, height, width]
    Y = tf.squeeze(Y, axis=-1)

    return Y

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
    # Make the function flexible to accepting h in multiple forms
    h_t = _as_col_tensor(h)
    m = h_t.get_shape().as_list()[0]
    m2 = m // 2

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns)
    X = tf.pad(X, [[0, 0], [m2, m2], [0, 0]], 'SYMMETRIC')

    Y = _conv_2d(X, h_t, strides=[1,1,1,1])

    return Y


def rowfilter(X, h):
    """Filter the rows of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [0 1].
    :param X: a tensor of images whose rows are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.
    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """
    # Make the function flexible to accepting h in multiple forms
    h_t = _as_row_tensor(h)
    m = h_t.get_shape().as_list()[1]
    m2 = m // 2

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns)
    X = tf.pad(X, [[0, 0], [0, 0], [m2, m2]], 'SYMMETRIC')

    Y = _conv_2d(X, h_t, strides=[1,1,1,1])

    return Y


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
    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    r, c = X.get_shape().as_list()[1:]
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    ha_t = _as_col_tensor(ha)
    hb_t = _as_col_tensor(hb)
    if ha_t.shape != hb_t.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    m = ha_t.get_shape().as_list()[0]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns).
    X = tf.pad(X, [[0, 0], [m, m], [0, 0]], 'SYMMETRIC')    

    # Take the odd and even columns of X
    X_odd = X[:,2:r+2*m-2:2,:]
    X_even =X[:,3:r+2*m-2:2,:]

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    a_rows = _conv_2d(X_odd, ha_t, strides=[1,2,1,1])
    b_rows = _conv_2d(X_even, hb_t, strides=[1,2,1,1])
    
    # Stack a_rows and b_rows (both of shape [Batch, r/4, c]) along the third
    # dimension to make a tensor of shape [Batch, r/4, 2, c].    
    Y = tf.cond(tf.reduce_sum(ha_t*hb_t) > 0,
                lambda: tf.stack([a_rows,b_rows],axis=2),
                lambda: tf.stack([b_rows,a_rows],axis=2))

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
    
    ha_t = _as_row_tensor(ha)
    hb_t = _as_row_tensor(hb)
    if ha_t.shape != hb_t.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    m = ha_t.get_shape().as_list()[1]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the rows).
    # SYMMETRIC extension means the edge sample is repeated twice, whereas
    # REFLECT only has the edge sample once    
    X = tf.pad(X, [[0, 0], [0, 0], [m, m]], 'SYMMETRIC')

    # Take the odd and even columns of X
    X_odd = X[:,:,2:c+2*m-2:2]
    X_even= X[:,:,3:c+2*m-2:2]

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    a_cols = _conv_2d(X_odd, ha_t, strides=[1,1,2,1])
    b_cols = _conv_2d(X_even, hb_t, strides=[1,1,2,1])
    
    # Stack a_cols and b_cols (both of shape [Batch, r, c/4]) along the fourth 
    # dimension to make a tensor of shape [Batch, r, c/4, 2].    
    Y = tf.cond(tf.reduce_sum(ha_t*hb_t) > 0,
                lambda: tf.stack([a_cols,b_cols],axis=3),
                lambda: tf.stack([b_cols,a_cols],axis=3))

    # Reshape result to be shape [Batch, r, c/2]. This reshaping interleaves
    # the columns
    Y = tf.reshape(Y, [-1, r, c2])   
    
    return Y


def colifilt(X, ha, hb):
    """ Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e `:math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext       left edge                      right edge       ext
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a

    The output is interpolated by two from the input sample rate and the
    results from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    
    r, c = X.get_shape().as_list()[1:]
    r2 = r // 2
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2')

    ha_t = _as_col_tensor(ha)
    hb_t = _as_col_tensor(hb)
    if ha_t.shape != hb_t.shape:
        raise ValueError('Shapes of ha and hb must be the same')
    
    m = ha_t.get_shape().as_list()[0]
    m2 = m // 2
    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    X = tf.pad(X, [[0, 0], [m2, m2], [0, 0]], 'SYMMETRIC')

    ha_odd_t  = ha_t[::2,:]
    ha_even_t = ha_t[1::2,:]
    hb_odd_t  = hb_t[::2,:]
    hb_even_t = hb_t[1::2,:]
    
    if m2 % 2 == 0:        
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        
        # Take the odd and even columns of X        
        X1,X2 = tf.cond(tf.reduce_sum(ha_t*hb_t) > 0,
                    lambda: (X[:,1:r+m-2:2,:], X[:,0:r+m-3:2,:]),
                    lambda: (X[:,0:r+m-3:2,:], X[:,1:r+m-2:2,:]))
        X3,X4 = tf.cond(tf.reduce_sum(ha_t*hb_t) > 0,
                    lambda: (X[:,3:r+m:2,:],   X[:,2:r+m-1:2,:]),
                    lambda: (X[:,2:r+m-1:2,:], X[:,3:r+m:2,:]))

        y1 = _conv_2d(X2, ha_even_t)
        y2 = _conv_2d(X1, hb_even_t)
        y3 = _conv_2d(X4, ha_odd_t)
        y4 = _conv_2d(X3, hb_odd_t)
        
    else:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        
        # Take the odd and even columns of X
        X1,X2 = tf.cond(tf.reduce_sum(ha_t*hb_t) > 0,
                    lambda: (X[:,2:r+m-1:2,:], X[:,1:r+m-2:2,:]),
                    lambda: (X[:,1:r+m-2:2,:], X[:,2:r+m-1:2,:])) 

        y1 = _conv_2d(X2, ha_odd_t)
        y2 = _conv_2d(X1, hb_odd_t)
        y3 = _conv_2d(X2, ha_even_t)
        y4 = _conv_2d(X1, hb_even_t)

    # Stack 4 tensors of shape [batch, r2, c] into one tensor [batch, r2, 4, c]
    Y = tf.stack([y1,y2,y3,y4], axis=2)
    
    # Reshape to be [batch, 2*4, c]. This interleaves the rows
    Y = tf.reshape(Y, [-1,2*r,c])
    
    return Y
