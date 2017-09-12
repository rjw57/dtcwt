from __future__ import absolute_import

import numpy as np

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.numpy.common import Pyramid as Pyramid_np
from dtcwt.utils import asfarray
from dtcwt.tf import Pyramid
from dtcwt.tf.lowlevel import coldfilt, colfilter, colifilt

try:
    import tensorflow as tf
    from tensorflow.python.framework import dtypes
    tf_dtypes = frozenset(
        [dtypes.float32, dtypes.float64, dtypes.int8, dtypes.int16,
         dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.qint8, dtypes.qint32,
         dtypes.quint8, dtypes.complex64, dtypes.complex128,
         dtypes.float32_ref, dtypes.float64_ref, dtypes.int8_ref,
         dtypes.int16_ref, dtypes.int32_ref, dtypes.int64_ref, dtypes.uint8_ref,
         dtypes.qint8_ref, dtypes.qint32_ref, dtypes.quint8_ref,
         dtypes.complex64_ref, dtypes.complex128_ref]
    )
except ImportError:
    # The lack of tensorflow will be caught by the low-level routines.
    pass

np_dtypes = frozenset(
    [np.dtype('float16'), np.dtype('float32'), np.dtype('float64'),
     np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
     np.dtype('int64'), np.dtype('uint8'), np.dtype('uint16'),
     np.dtype('uint32'), np.dtype('complex64'), np.dtype('complex128')]
)


class Transform1d(object):
    """
    An implementation of the 1D DT-CWT in Tensorflow.

    :param biort: Level 1 wavelets to use. See :py:func:`dtcwt.coeffs.biort`.
    :param qshift: Level >= 2 wavelets to use. See
        :py:func:`dtcwt.coeffs.qshift`.

    .. note::

        Calling the methods in this class with different inputs will slightly
        vary the results. If you call the
        :py:meth:`~dtcwt.tf.Transform1d.forward` or
        :py:meth:`~dtcwt.tf.Transform1d.forward_channels` methods with a numpy
        array, they load this array into a :py:class:`tf.Variable` and create
        the graph. Subsequent calls to :py:attr:`dtcwt.tf.Pyramid.lowpass` or
        other attributes in the pyramid will create a session and evaluate these
        parameters.  If the above methods are called with a tensorflow variable
        or placeholder, these will be used to create the graph. As such, to
        evaluate the results, you will need to look at the
        :py:attr:`dtcwt.tf.Pyramid.lowpass_op` attribute (calling the `lowpass`
        attribute will try to evaluate the graph with no initialized variables
        and likely result in a runtime error).

        The behaviour is similar for the
        :py:meth:`~dtcwt.tf.Transform1d.inverse` and
        :py:meth:`~dtcwt.tf.Transform1d.inverse_channels` methods, except these
        return an array, rather than a Pyramid style class. If a
        :py:class:`dtcwt.tf.Pyramid` was created by calling the forward methods
        with a numpy array, providing this pyramid to the inverse methods will
        return a numpy array. If however a :py:class:`dtcwt.tf.Pyramid` was
        created by calling the forward methods with a tensorflow variable, the
        result from calling the inverse methods will also be a tensorflow
        variable.
    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        self.biort = biort
        self.qshift = qshift

    def forward(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT decompostion on a 1D column vector *X* (or on
        the columns of a matrix *X*).

        Can provide the forward transform with either an np array (naive usage),
        or a tensorflow variable or placeholder (designed usage). To transform
        batches of vectors, use the :py:meth:`forward_channels` method.

        :param X: 1D real array or 2D real array whose columns are to be
            transformed.
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.tf.Pyramid` object representing the
            transform result.

        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
        interpreted as tuples of vectors giving filter coefficients. In the
        *biort* case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case,
        this should be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Sep 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Check if a numpy array was provided
        numpy = False
        try:
            dtype = X.dtype
        except AttributeError:
            X = asfarray(X)
            dtype = X.dtype

        if dtype in np_dtypes:
            numpy = True
            # Need this because colfilter and friends assumes input is 2d
            if len(X.shape) == 1:
                X = np.atleast_2d(X).T
            X = tf.Variable(X, dtype=tf.float32, trainable=False)
        elif dtype in tf_dtypes:
            if len(X.get_shape().as_list()) == 1:
                X = tf.expand_dims(X, axis=-1)
        else:
            raise ValueError('I cannot handle the variable you have ' +
                             'provided of type ' + str(X.dtype) + '. ' +
                             'Inputs should be a numpy or tf array')

        X_shape = tuple(X.get_shape().as_list())
        size = '{}'.format(X_shape[0])
        name = 'dtcwt_fwd_{}'.format(size)
        if len(X_shape) == 2:
            # Need to make it a batch for tensorflow
            X = tf.expand_dims(X, axis=0)
        elif len(X_shape) >= 3:
            raise ValueError(
                'The entered variable has too many ' +
                'dimensions - ' + str(X_shape) + '.')

        # Do the forward transform
        with tf.variable_scope(name):
            Yl, Yh, Yscale = self._forward_ops(X, nlevels)

        Yl = Yl[0]
        Yh = tuple(x[0] for x in Yh)
        Yscale = tuple(x[0] for x in Yscale)

        if include_scale:
            return Pyramid(Yl, Yh, Yscale, numpy)
        else:
            return Pyramid(Yl, Yh, None, numpy)

    def forward_channels(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT decompostion on a 3D array *X*.

        Can provide the forward transform with either an np array (naive usage),
        or a tensorflow variable or placeholder (designed usage).

        :param X: 3D real array. Batch of matrices whose columns are to be
            transformed (i.e. the second dimension).
        :param nlevels: Number of levels of wavelet decomposition

        :returns: A :py:class:`dtcwt.tf.Pyramid` object representing the
            transform result.

        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`biort` or :py:func:`qshift` functions. Otherwise, they are
        interpreted as tuples of vectors giving filter coefficients. In the
        *biort* case, this should be (h0o, g0o, h1o, g1o). In the *qshift* case,
        this should be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Sep 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Check if a numpy array was provided
        numpy = False
        try:
            dtype = X.dtype
        except AttributeError:
            X = asfarray(X)
            dtype = X.dtype

        if dtype in np_dtypes:
            numpy = True
            if len(X.shape) != 3:
                raise ValueError(
                    'Incorrect input shape for the forward_channels ' +
                    'method ' + str(X.shape) + '. For Inputs of 1 or 2 ' +
                    'dimensions, use the forward method.')
            # Need this because colfilter and friends assumes input is 2d
            X = tf.Variable(X, dtype=tf.float32, trainable=False)
        elif dtype in tf_dtypes:
            X_shape = X.get_shape().as_list()
            if len(X.get_shape().as_list()) != 3:
                raise ValueError(
                    'Incorrect input shape for the forward_channels ' +
                    'method ' + str(X_shape) + '. For Inputs of 1 or 2 ' +
                    'dimensions, use the forward method.')
        else:
            raise ValueError('I cannot handle the variable you have ' +
                             'provided of type ' + str(X.dtype) + '. ' +
                             'Inputs should be a numpy or tf array')

        X_shape = tuple(X.get_shape().as_list())
        size = '{}'.format(X_shape[1])
        name = 'dtcwt_fwd_{}'.format(size)

        # Do the forward transform
        with tf.variable_scope(name):
            Yl, Yh, Yscale = self._forward_ops(X, nlevels)

        if include_scale:
            return Pyramid(Yl, Yh, Yscale, numpy)
        else:
            return Pyramid(Yl, Yh, None, numpy)

    def inverse(self, pyramid, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 1D
        reconstruction.

        :param pyramid: A :py:class:`dtcwt.Pyramid`-like object containing
            the transformed signal.
        :param gain_mask: Gain to be applied to each subband.

        :returns: Reconstructed real array. Will be a tf Variable if the Pyramid
            was made with tf inputs, otherwise a numpy array.


        The *l*-th element of *gain_mask* is gain for wavelet subband at level
        l.  If gain_mask[l] == 0, no computation is performed for band *l*.
        Default *gain_mask* is all ones. Note that *l* is 0-indexed.

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # A tensorflow object was provided
        numpy = False
        if isinstance(pyramid, Pyramid):
            Yl = pyramid.lowpass_op
            Yh = pyramid.highpasses_ops
            numpy = pyramid.numpy

        # Check if a numpy pyramid was provided
        elif isinstance(pyramid, Pyramid_np) or \
                hasattr(pyramid, 'lowpass') and hasattr(pyramid, 'highpasses'):
            numpy = True
            Yl, Yh = pyramid.lowpass, pyramid.highpasses
            Yl = tf.Variable(Yl, trainable=False, dtype=tf.float32)
            Yh = tuple(
                tf.Variable(level, trainable=False, dtype=tf.complex64)
                for level in Yh)
        else:
            raise ValueError(
                'Unknown pyramid provided to inverse transform')

        # Need to make sure it has at least 3 dimensions for tensorflow
        Yl_shape = tuple(Yl.get_shape().as_list())
        if len(Yl_shape) == 2:
            Yl = tf.expand_dims(Yl, axis=0)
            Yh = tuple(tf.expand_dims(x, axis=0) for x in Yh)
        elif len(Yl_shape) >= 3:
            raise ValueError(
                'The entered variables have too many ' +
                'dimensions - ' + str(Yl_shape) + '. For batches of ' +
                'images with multiple channels (i.e. 3 or 4 dimensions), ' +
                'please either enter each channel separately, or use ' +
                'the inverse_channels method.')

        # Do the inverse transform
        s = Yl.get_shape().as_list()[1]
        nlevels = len(Yh)
        size = '{}_up_{}'.format(s, nlevels)
        name = 'dtcwt_inv_{}'.format(size)
        with tf.variable_scope(name):
            X = self._inverse_ops(Yl, Yh, gain_mask)

        # Chop off the first dimension
        X = X[0]

        # Return a 1d vector or a column vector
        if X.get_shape().as_list()[1] == 1:
            X = X[:,0]

        if numpy:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                X = sess.run(X)

        return X

    def inverse_channels(self, pyramid, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 1D
        reconstruction on a 3D array of signals. The inverse is done on the
        second dimension of these.

        This is designed to work after calling the
        :py:meth:`~dtcwt.tf.Transform1d.forward_channels` method.

        :param pyramid: A :py:class:`dtcwt.Pyramid`-like object containing
            the transformed signal. The lowpass signal in the pyramid should be
            a 3D array to use this method.
        :param gain_mask: Gain to be applied to each subband.

        :returns: Reconstructed array. Will be a tf Variable if the Pyramid was
            made with tf inputs, otherwise a numpy array.

        The *l*-th element of *gain_mask* is gain for wavelet subband at level
        l.  If gain_mask[l] == 0, no computation is performed for band *l*.
        Default *gain_mask* is all ones. Note that *l* is 0-indexed.

        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # A tensorflow object was provided
        numpy = False
        if isinstance(pyramid, Pyramid):
            Yl = pyramid.lowpass_op
            Yl_shape = Yl.get_shape().as_list()
            if len(Yl_shape) != 3:
                raise ValueError(
                    'Incorrect input shape for the forward_channels ' +
                    'method ' + str(Yl_shape) + '. For Inputs of 1 or 2 ' +
                    'dimensions, use the forward method.')
            Yh = pyramid.highpasses_ops
            numpy = pyramid.numpy

        # Check if a numpy pyramid was provided
        elif isinstance(pyramid, Pyramid_np) or \
                hasattr(pyramid, 'lowpass') and hasattr(pyramid, 'highpasses'):
            numpy = True
            Yl, Yh = pyramid.lowpass, pyramid.highpasses
            if len(Yl.shape) != 3:
                raise ValueError(
                    'Incorrect input shape for the forward_channels ' +
                    'method ' + str(Yl.shape) + '. For Inputs of 1 or 2 ' +
                    'dimensions, use the forward method.')

            Yl = tf.Variable(Yl, trainable=False, dtype=tf.float32)
            Yh = tuple(
                tf.Variable(level, trainable=False, dtype=tf.complex64)
                for level in Yh)
        else:
            raise ValueError(
                'Unknown pyramid provided to inverse transform')

        # Do the inverse transform
        s = Yl.get_shape().as_list()[1]
        nlevels = len(Yh)
        size = '{}_up_{}'.format(s, nlevels)
        name = 'dtcwt_inv_{}'.format(size)
        with tf.variable_scope(name):
            X = self._inverse_ops(Yl, Yh, gain_mask)

        if numpy:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                X = sess.run(X)

        return X

    def _forward_ops(self, X, nlevels=3):
        """ Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        For column inputs, we still need the input shape to be 3D, but with 1 as
        the last dimension.

        :param X: 3D real array of size [batch, h, w]
        :param nlevels: Number of levels of wavelet decomposition
        :param extended: True if a singleton dimension was added at the
            beginning of the input. Signal to remove afterwards.

        :returns: A tuple of Yl, Yh, Yscale
        """
        biort = self.biort
        qshift = self.qshift

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        # Check the shape and form of the input
        if X.dtype not in tf_dtypes:
            raise ValueError('X needs to be a tf variable or placeholder')

        original_size = X.get_shape().as_list()[1:]

        # ############################ Resize #################################
        # The next few lines of code check to see if the image is odd in size,
        # if so an extra ... row/column will be added to the bottom/right of the
        # image
        #  initial_row_extend = 0
        #  initial_col_extend = 0
        # If the row count of X is not divisible by 2 then we need to
        # extend X by adding a row at the bottom
        if original_size[0] % 2 != 0:
            #  X = tf.pad(X, [[0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
            raise ValueError('Size of input X must be a multiple of 2')

        #  extended_size = X.get_shape().as_list()[1:]

        if nlevels == 0:
            return X, (), ()

        # ########################### Initialise ###############################
        Yh = [None, ] * nlevels
        # This is only required if the user specifies a third output
        # component.
        Yscale = [None, ] * nlevels

        # ############################ Level 1 #################################
        # Uses the biorthogonal filters
        if nlevels >= 1:
            # Do odd top-level filters on cols.
            Hi = colfilter(X, h1o)
            Lo = colfilter(X, h0o)

            # Convert Hi to complex form by taking alternate rows
            Yh[0] = tf.cast(Hi[:,::2,:], tf.complex64) + \
                1j*tf.cast(Hi[:,1::2,:], tf.complex64)
            Yscale[0] = Lo

        # ############################ Level 2+ ################################
        # Uses the qshift filters
        for level in xrange(1, nlevels):
            # If the row count of Lo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra rows to make it so
            if Lo.get_shape().as_list()[1] % 4 != 0:
                Lo = tf.pad(Lo, [[0, 0], [1, 1], [0, 0]], 'SYMMETRIC')

            # Do even Qshift filters on cols.
            Hi = coldfilt(Lo, h1b, h1a)
            Lo = coldfilt(Lo, h0b, h0a)

            # Convert Hi to complex form by taking alternate rows
            Yh[level] = tf.cast(Hi[:,::2,:], tf.complex64) + \
                1j * tf.cast(Hi[:,1::2,:], tf.complex64)
            Yscale[level] = Lo

        Yl = Lo

        return Yl, tuple(Yh), tuple(Yscale)

    def _inverse_ops(self, Yl, Yh, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 1D
        reconstruction.

        :param Yl: The lowpass output from a forward transform. Should be a
            tensorflow variable.
        :param Yh: The tuple of highpass outputs from a forward transform.
            Should be tensorflow variables.
        :param gain_mask: Gain to be applied to each subband.

        :returns: A tf.Variable holding the output

        The *l*-th element of *gain_mask* is gain for wavelet subband at level
        l.  If gain_mask[l] == 0, no computation is performed for band *l*.
        Default *gain_mask* is all ones. Note that *l* is 0-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Sep 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        # Which wavelets are to be used?
        biort = self.biort
        qshift = self.qshift
        a = len(Yh)  # No of levels.

        if gain_mask is None:
            gain_mask = np.ones(a)  # Default gain_mask.
        gain_mask = np.array(gain_mask)

        # Try to load coefficients if biort is a string parameter
        try:
            h0o, g0o, h1o, g1o = _biort(biort)
        except TypeError:
            h0o, g0o, h1o, g1o = biort

        # Try to load coefficients if qshift is a string parameter
        try:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        except TypeError:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = qshift

        level = a-1   # No of levels = no of rows in L.
        if level < 0:
            # if there are no levels in the input, just return the Yl value
            return Yl

        # Reconstruct levels 2 and above in reverse order.
        Lo = Yl
        while level >= 1:
            Hi = c2q1d(Yh[level]*gain_mask[level])
            Lo = colifilt(Lo, g0b, g0a) + colifilt(Hi, g1b, g1a)

            # If Lo is not the same length as the next Therefore we have to clip
            # Lo so it is the same height as the next Yh. Yh => t1 was extended.
            Lo_shape = Lo.get_shape().as_list()
            next_shape = Yh[level-1].get_shape().as_list()
            if Lo_shape[1] != 2 * next_shape[1]:
                Lo = Lo[:,1:-1]
                Lo_shape = Lo.get_shape().as_list()

            # Check the row shapes across the entire matrix
            if (np.any(np.asanyarray(Lo_shape[1:]) !=
                       np.asanyarray(next_shape[1:] * np.array((2,1))))):
                raise ValueError('Yh sizes are not valid for DTWAVEIFM')

            level -= 1

        # Reconstruct level 1.
        if level == 0:
            Hi = c2q1d(Yh[level]*gain_mask[level])
            Z = colfilter(Lo,g0o) + colfilter(Hi,g1o)

        return Z


# =============================================================================
#              **********      INTERNAL FUNCTION    **********
# =============================================================================
def c2q1d(x):
    """ An internal function to convert a 1D Complex vector back to a real
    array,  which is twice the height of x.
    """
    # Input has shape [batch, r, c, 2]
    r, c = x.get_shape().as_list()[1:3]
    x1 = tf.real(x)
    x2 = tf.imag(x)
    # Stack 2 inputs of shape [batch, r, c] to [batch, r, 2, c]
    y = tf.stack([x1, x2], axis=-2)
    # Reshaping interleaves the results
    y = tf.reshape(y, [-1, 2 * r, c])

    return y

# vim:sw=4:sts=4:et
