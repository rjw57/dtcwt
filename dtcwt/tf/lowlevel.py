from __future__ import absolute_import

try:
    import tensorflow as tf
    _HAVE_TF = True
except ImportError:
    _HAVE_TF = False

from dtcwt.utils import as_column_vector
import numpy as np


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
    """
    Perform 2d convolution in tensorflow.

    X will to be manipulated to be of shape [batch, height, width, ch],
    and h to be of shape [height, width, ch, num]. This function does the
    necessary reshaping before calling the conv2d function, and does the
    reshaping on the output, returning Y of shape [batch, height, width]
    """

    # Check the shape of X is what we expect
    if len(X.shape) != 3:
        raise ValueError('X needs to be of shape [batch, height, width] ' +
                         'for conv_2d')

    # Check the shape of h is what we expect
    if len(h.shape) != 2:
        raise ValueError('Filter inputs must only have height and width ' +
                         'for conv_2d')

    # Add in the unit dimensions for conv
    X = tf.expand_dims(X, axis=-1)
    h = tf.expand_dims(tf.expand_dims(h, axis=-1),axis=-1)

    # Have to reverse h as tensorflow 2d conv is actually cross-correlation
    h = tf.reverse(h, axis=[0,1])
    Y = tf.nn.conv2d(X, h, strides=strides, padding='VALID')

    # Remove the final dimension, returning a result of shape
    # [batch, height, width]
    Y = tf.squeeze(Y, axis=-1)

    return Y


def _conv_2d_transpose(X, h, out_shape, strides=[1,1,1,1]):
    """
    Perform 2d transpose convolution in tensorflow.

    X will to be manipulated to be of shape [batch, height, width, ch], and h to
    be of shape [height, width, ch, num]. This function does the necessary
    reshaping before calling the conv2d function, and does the reshaping on the
    output, returning Y of shape [batch, height, width]
    """

    # Check the shape of X is what we expect
    if len(X.shape) != 3:
        raise ValueError('X needs to be of shape [batch, height, width] ' +
                         'for conv_2d')
    # Check the shape of h is what we expect
    if len(h.shape) != 2:
        raise ValueError('Filter inputs must only have height and width ' +
                         'for conv_2d')

    # Add in the unit dimensions for conv
    X = tf.expand_dims(X, axis=-1)
    h = tf.expand_dims(tf.expand_dims(h, axis=-1),axis=-1)

    # Have to reverse h as tensorflow 2d conv is actually cross-correlation
    h = tf.reverse(h, axis=[0,1])
    # Transpose h as we will be using the transpose convolution
    h = tf.transpose(h, perm=[1, 0, 2, 3])

    Y = tf.nn.conv2d(X, h, output_shape=out_shape, strides=strides,
                     padding='VALID')

    # Remove the final dimension, returning a result of shape
    # [batch, height, width]
    Y = tf.squeeze(Y, axis=-1)

    return Y


def _tf_pad(x, szs, padding='SYMMETRIC'):
    """
    Tensorflow can't handle padding by more than the dimension of the image.
    This wrapper allows us to build padding up successively.
    """
    def get_size(x):
        # Often the batch will be None. Convert these to 0s
        x_szs = x.get_shape().as_list()
        x_szs = [0 if val is None else val for val in x_szs]
        return x_szs

    x_szs = get_size(x)
    gt = [[sz[0] > x_sz, sz[1] > x_sz] for sz,x_sz in zip(szs, x_szs)]
    while np.any(gt):
        # This creates an intermediate padding amount that will bring in
        # dimensions that are too big by the size of x.
        szs_step = np.int32(gt) * np.stack([x_szs, x_szs], axis=-1)
        x = tf.pad(x, szs_step, padding)
        szs = szs - szs_step
        x_szs = get_size(x)
        gt = [[sz[0] > x_sz, sz[1] > x_sz] for sz,x_sz in zip(szs, x_szs)]

    # Pad by the remaining amount
    x = tf.pad(x, szs, 'SYMMETRIC')
    return x


def colfilter(X, h, align=False):
    """
    Filter the columns of image *X* using filter vector *h*, without decimation.

    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :param align: If true, then will have Y keep the same output shape as X,
        even if h has even length. Makes no difference if len(h) is odd.

    :returns Y: the filtered image.

    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.
    If len(h) is even, each output sample is aligned with the mid point of
    each pair of input samples, and Y.shape = X.shape + [1 0].

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
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
    if m % 2 == 0 and align:
        X = _tf_pad(X, [[0, 0], [m2 - 1, m2], [0, 0]], 'SYMMETRIC')
    else:
        X = _tf_pad(X, [[0, 0], [m2, m2], [0, 0]], 'SYMMETRIC')

    Y = _conv_2d(X, h_t, strides=[1,1,1,1])

    return Y


def rowfilter(X, h, align=False):
    """
    Filter the rows of image *X* using filter vector *h*, without decimation.

    :param X: a tensor of images whose rows are to be filtered
    :param h: the filter coefficients.
    :param align: If true, then will have Y keep the same output shape as X,
        even if h has even length. Makes no difference if len(h) is odd.

    :returns Y: the filtered image.

    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.
    If len(h) is even, each output sample is aligned with the mid point of each
    pair of input samples, and Y.shape = X.shape + [0 1].

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
    if m % 2 == 0 and align:
        X = _tf_pad(X, [[0, 0], [0, 0], [m2 - 1, m2]], 'SYMMETRIC')
    else:
        X = _tf_pad(X, [[0, 0], [0, 0], [m2, m2]], 'SYMMETRIC')

    Y = _conv_2d(X, h_t, strides=[1,1,1,1])

    return Y


def coldfilt(X, ha, hb, no_decimate=False):
    """
    Filter the columns of image X using the two filters ha and hb =
    reverse(ha).

    :param X: The input, of size [batch, h, w]
    :param ha: Filter to be used on the odd samples of x.
    :param hb: Filter to bue used on the even samples of x.
    :param no_decimate: If true, keep the same input size

    Both filters should be even length, and h should be approx linear
    phase with a quarter sample (i.e. an :math:`e^{j \pi/4}`) advance from
    its mid pt (i.e. :math:`|h(m/2)| > |h(m/2 + 1)|`)::

                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    :raises ValueError if the number of rows in X is not a multiple of 4, the
        length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    r, c = X.get_shape().as_list()[1:]
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.get_shape().as_list()))

    ha_t = _as_col_tensor(ha)
    hb_t = _as_col_tensor(hb)
    if ha_t.shape != hb_t.shape:
        raise ValueError('Shapes of ha and hb must be the same\n' +
                         'ha was {}, hb was {}'.format(ha_t.shape, hb_t.shape))

    m = ha_t.get_shape().as_list()[0]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even\n' +
                         'ha was {}, hb was {}'.format(ha_t.shape, hb_t.shape))

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    rows = r2
    if no_decimate:
        pass

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns).
    X = _tf_pad(X, [[0, 0], [m, m], [0, 0]], 'SYMMETRIC')

    # Take the odd and even columns of X
    X_odd = X[:, 2:r + 2 * m - 2:2, :]
    X_even = X[:, 3:r + 2 * m - 2:2, :]

    a_rows = _conv_2d(X_odd, ha_t, strides=[1,2,1,1])
    b_rows = _conv_2d(X_even, hb_t, strides=[1,2,1,1])

    # Stack a_rows and b_rows (both of shape [Batch, r/4, c]) along the third
    # dimension to make a tensor of shape [Batch, r/4, 2, c].
    Y = tf.cond(tf.reduce_sum(ha_t * hb_t) > 0,
                lambda: tf.stack([a_rows, b_rows],axis=2),
                lambda: tf.stack([b_rows, a_rows],axis=2))

    # Reshape result to be shape [Batch, r/2, c]. This reshaping interleaves
    # the columns
    Y = tf.reshape(Y, [-1, rows, c])

    return Y


def rowdfilt(X, ha, hb, no_decimate=False):
    """
    Filter the rows of image X using the two filters ha and hb = reverse(ha).

    :param X: The input, of size [batch, h, w]
    :param ha: Filter to be used on the odd samples of x.
    :param hb: Filter to bue used on the even samples of x.
    :param no_decimate: If true, keep the same input size

    Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`)::

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

    :raises ValueError if the number of columns in X is not a multiple of 4, the
        length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    r, c = X.get_shape().as_list()[1:]
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.get_shape().as_list()))

    ha_t = _as_row_tensor(ha)
    hb_t = _as_row_tensor(hb)
    if ha_t.shape != hb_t.shape:
        raise ValueError('Shapes of ha and hb must be the same\n' +
                         'ha was {}, hb was {}'.format(ha_t.shape, hb_t.shape))

    m = ha_t.get_shape().as_list()[1]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even\n' +
                         'ha was {}, hb was {}'.format(ha_t.shape, hb_t.shape))

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the rows).
    # SYMMETRIC extension means the edge sample is repeated twice, whereas
    # REFLECT only has the edge sample once
    X = _tf_pad(X, [[0, 0], [0, 0], [m, m]], 'SYMMETRIC')

    # Take the odd and even columns of X
    X_odd = X[:,:,2:c + 2 * m - 2:2]
    X_even = X[:,:,3:c + 2 * m - 2:2]

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    cols = c2
    if no_decimate:
        pass

    a_cols = _conv_2d(X_odd, ha_t, strides=[1,1,2,1])
    b_cols = _conv_2d(X_even, hb_t, strides=[1,1,2,1])

    # Stack a_cols and b_cols (both of shape [Batch, r, c/4]) along the fourth
    # dimension to make a tensor of shape [Batch, r, c/4, 2].
    Y = tf.cond(tf.reduce_sum(ha_t * hb_t) > 0,
                lambda: tf.stack([a_cols, b_cols], axis=3),
                lambda: tf.stack([b_cols, a_cols], axis=3))

    # Reshape result to be shape [Batch, r, c/2]. This reshaping interleaves
    # the columns
    Y = tf.reshape(Y, [-1, r, cols])

    return Y


def colifilt(X, ha, hb, no_decimate=False):
    """
    Filter the columns of image X using the two filters ha and hb =
    reverse(ha).

    :param X: The input, of size [batch, h, w]
    :param ha: Filter to be used on the odd samples of x.
    :param hb: Filter to bue used on the even samples of x.
    :param no_decimate: Not implemented yet

    Both filters should be even length, and h should be approx linear
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

    # A quick hack to handle undecimated inputs. Simply take every second sample
    # as if it had been decimated.
    r, c = X.get_shape().as_list()[1:]
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2.\n' +
                         'X was {}'.format(X.get_shape().as_list()))

    ha_t = _as_col_tensor(ha)
    hb_t = _as_col_tensor(hb)
    if ha_t.shape != hb_t.shape:
        raise ValueError('Shapes of ha and hb must be the same.\n' +
                         'ha was {}, hb was {}'.format(ha_t.shape, hb_t.shape))

    m = ha_t.get_shape().as_list()[0]
    m2 = m // 2
    if ha_t.get_shape().as_list()[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even.\n' +
                         'ha was {}, hb was {}'.format(ha_t.shape, hb_t.shape))

    X = _tf_pad(X, [[0, 0], [m2, m2], [0, 0]], 'SYMMETRIC')

    ha_odd_t = ha_t[::2,:]
    ha_even_t = ha_t[1::2,:]
    hb_odd_t = hb_t[::2,:]
    hb_even_t = hb_t[1::2,:]

    if m2 % 2 == 0:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.

        # Take the odd and even columns of X
        X1, X2 = tf.cond(
            tf.reduce_sum(ha_t * hb_t) > 0,
            lambda: (X[:, 1:r + m - 2:2, :], X[:, 0:r + m - 3:2, :]),
            lambda: (X[:, 0:r + m - 3:2, :], X[:, 1:r + m - 2:2, :]))
        X3, X4 = tf.cond(
            tf.reduce_sum(ha_t * hb_t) > 0,
            lambda: (X[:, 3:r + m:2, :], X[:, 2:r + m - 1:2, :]),
            lambda: (X[:, 2:r + m - 1:2, :], X[:, 3:r + m:2, :]))

        y1 = _conv_2d(X2, ha_even_t)
        y2 = _conv_2d(X1, hb_even_t)
        y3 = _conv_2d(X4, ha_odd_t)
        y4 = _conv_2d(X3, hb_odd_t)

    else:
        # m/2 is odd, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.

        # Take the odd and even columns of X
        X1, X2 = tf.cond(
            tf.reduce_sum(ha_t * hb_t) > 0,
            lambda: (X[:, 2:r + m - 1:2, :], X[:, 1:r + m - 2:2, :]),
            lambda: (X[:, 1:r + m - 2:2, :], X[:, 2:r + m - 1:2, :]))

        y1 = _conv_2d(X2, ha_odd_t)
        y2 = _conv_2d(X1, hb_odd_t)
        y3 = _conv_2d(X2, ha_even_t)
        y4 = _conv_2d(X1, hb_even_t)

    # Stack 4 tensors of shape [batch, r2, c] into one tensor [batch, r2, 4, c]
    Y = tf.stack([y1,y2,y3,y4], axis=2)

    # Reshape to be [batch, r * 2, c]. This interleaves the rows
    Y = tf.reshape(Y, [-1,2*r,c])

    return Y
