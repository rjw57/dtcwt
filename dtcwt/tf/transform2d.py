from __future__ import absolute_import

import numpy as np
import logging
import warnings

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import asfarray
from dtcwt.tf import Pyramid
from dtcwt.numpy import Pyramid as Pyramid_np

from dtcwt.tf.lowlevel import coldfilt, rowdfilt, rowfilter, colfilter, colifilt

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


def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT,
               include_scale=False):
    t = Transform2d(biort=biort, qshift=qshift)
    r = t.forward(X, nlevels=nlevels, include_scale=include_scale)
    if include_scale:
        return r.lowpass, r.highpasses, r.scales
    else:
        return r.lowpass, r.highpasses


def dtwaveifm2(Yl, Yh, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT,
               gain_mask=None):
    t = Transform2d(biort=biort, qshift=qshift)
    r = t.inverse(Pyramid_np(Yl, Yh), gain_mask=gain_mask)
    return r


class Transform2d(object):
    """
    An implementation of the 2D DT-CWT via Tensorflow.

    :param biort: The biorthogonal wavelet family to use.
    :param qshift: The quarter shift wavelet family to use.

    .. note::

        *biort* and *qshift* are the wavelets which parameterise the transform.
        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`dtcwt.coeffs.biort` or :py:func:`dtcwt.coeffs.qshift`
        functions.  Otherwise, they are interpreted as tuples of vectors giving
        filter coefficients. In the *biort* case, this should be (h0o, g0o, h1o,
        g1o). In the *qshift* case, this should be (h0a, h0b, g0a, g0b, h1a,
        h1b, g1a, g1b).

    Creating an object of this class loads the necessary filters onto the
    tensorflow graph. A subsequent call to :py:func:`Transform2d.forward` with
    an image (or placeholder) will create a forward transform for an input of
    the image's size. You can evaluate the resulting ops several times feeding
    different images into the placeholder *assuming* they have the same
    resolution. For a different resolution image, call the
    :py:func:`Transform2d.forward` function again.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
    .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
    """

    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        try:
            self.biort = _biort(biort)
        except TypeError:
            self.biort = biort

        # Load quarter sample shift wavelets
        try:
            self.qshift = _qshift(qshift)
        except TypeError:
            self.qshift = qshift
        # Use our own graph when the user calls forward with numpy arrays
        self.np_graph = tf.Graph()
        self.forward_graphs = {}
        self.inverse_graphs = {}

    def _find_forward_graph(self, shape):
        """ See if we can reuse an old graph for the forward transform """
        find_key = '{}x{}'.format(shape[0], shape[1])
        for key, val in self.forward_graphs.items():
            if find_key == key:
                return val
        return None

    def _add_forward_graph(self, p_ops, shape):
        """ Keep record of the pyramid so we can use it later if need be """
        find_key = '{}x{}'.format(shape[0], shape[1])
        self.forward_graphs[find_key] = p_ops

    def _find_inverse_graph(self, Lo_shape, nlevels):
        """ See if we can reuse an old graph for the inverse transform """
        find_key = '{}x{}'.format(Lo_shape[0], Lo_shape[1])
        for key, val in self.forward_graphs.items():
            if find_key == key:
                return val
        return None

    def _add_inverse_graph(self, p_ops, Lo_shape, nlevels):
        """ Keep record of the pyramid so we can use it later if need be """
        find_key = '{}x{} up {}'.format(Lo_shape[0], Lo_shape[1], nlevels)
        self.inverse_graphs[find_key] = p_ops

    def forward(self, X, nlevels=3, include_scale=False):
        """
        Perform a forward transform on an image.

        Can provide the forward transform with either an np array (naive
        usage), or a tensorflow variable or placeholder (designed usage).

        :param ndarray X: Input image which you wish to transform. Can be a
            numpy array, tensorflow Variable or tensorflow placeholder. See
            comments below.
        :param int nlevels: Number of levels of the dtcwt transform to
            calculate.
        :param bool include_scale: Whether or not to return the lowpass results
            at each scale of the transform, or only at the highest scale (as is
            custom for multi-resolution analysis)

        :returns: A :py:class:`Pyramid` like object

        .. note::

            If a numpy array is provided, the forward function will create a
            tensorflow variable to hold the input image, and then create the
            graph of the right size to match the input, and then feed the
            input into the graph and evaluate it.  This operation will
            return a :py:class:`Pyramid` object similar to how running
            the numpy version would.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
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
            X = np.atleast_2d(X)
            X = tf.Variable(X, dtype=tf.float32, trainable=False)

        if X.dtype not in tf_dtypes:
            raise ValueError('I cannot handle the variable you have ' +
                             'provided of type ' + str(X.dtype) + '. ' +
                             'Inputs should be a numpy or tf array')

        X_shape = tuple(X.get_shape().as_list())
        extended = False
        if len(X_shape) == 2:
            # Need to make it a batch for tensorflow
            X = tf.expand_dims(X, axis=0)
            extended = True
        elif len(X_shape) == 3:
            if X_shape[2] == 3 and X_shape[1] != 3:
                warnings.warn('It looks like you may have entered an RGB ' +
                              'image of shape ' + str(X_shape) + '. The ' +
                              'tf backend can handle batches of images, ' +
                              'but needs the batch to be the zeroth ' +
                              'dimension.')
        elif len(X_shape) > 3:
            raise ValueError(
                'The entered variable has too many ' +
                'dimensions - ' + str(X_shape) + '. For batches of ' +
                'images with multiple channels (i.e. 4 dimensions), ' +
                'please either enter each channel separately, or use ' +
                'the forward_channels method.')

        X_shape = tuple(X.get_shape().as_list())
        original_size = X_shape[1:]
        size = '{}x{}'.format(original_size[0], original_size[1])
        name = 'dtcwt_fwd_{}'.format(size)
        with tf.name_scope(name):
            Yl, Yh, Yscale = self._forward_ops(X, nlevels)

        if extended:
            Yl = Yl[0]
            Yh = tuple(x[0] for x in Yh)
            Yscale = tuple(x[0] for x in Yscale)

        if include_scale:
            return Pyramid(Yl, Yh, Yscale, numpy)
        else:
            return Pyramid(Yl, Yh, None, numpy)

    def forward_channels(self, X, nlevels=3, include_scale=False,
                         data_format="nhwc"):
        """ Perform a forward transform on an image with multiple channels.

        Will perform the DTCWT independently on each channel.

        :param X: Input image which you wish to transform.
        :param int nlevels: Number of levels of the dtcwt transform to
            calculate.
        :param bool include_scale: Whether or not to return the lowpass results
            at each scale of the transform, or only at the highest scale (as is
            custom for multiresolution analysis)
        :param str data_format: An optional string of the form "nchw" or "nhwc",
            specifying the data format of the input. If format is "nchw" (the
            default), then data is in the form [batch, channels, h, w]. If the
            format is "nhwc", then the data is in the form [batch, h, w, c].

        :returns: Yl - the lowpass output and the final scale.
        :returns: Yh - the highpass outputs.
        :returns: Yscale - the lowpass output at intermediate scales.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
        """
        data_format = data_format.lower()
        if data_format != "nchw" and data_format != "nhwc":
            raise ValueError('The data format must be either "ncwh" or ' +
                             '"nhwc", not {}'.format(data_format))

        try:
            dtype = X.dtype
        except AttributeError:
            X = asfarray(X)
            dtype = X.dtype

        numpy = False
        if dtype in np_dtypes:
            numpy = True
            X = np.atleast_2d(X)
            X = tf.Variable(X, dtype=tf.float32, trainable=False)

        if X.dtype not in tf_dtypes:
            raise ValueError('I cannot handle the variable you have ' +
                             'provided of type ' + str(X.dtype) + '. ' +
                             'Inputs should be a numpy or tf array.')

        X_shape = X.get_shape().as_list()
        if len(X_shape) != 4:
            raise ValueError(
                'The entered variable has incorrect shape - ' +
                str(X_shape) + '.  It must have 4 dimensions. For 2 or 3 ' +
                'dimensioned input, use the forward method.')

        # Move all of the channels into the batch dimension for the
        # input.  This may involve transposing, depending on the data
        # format
        with tf.variable_scope('ch_to_batch'):
            s = X.get_shape().as_list()[1:]
            size = '{}x{}'.format(s[0], s[1])
            name = 'dtcwt_fwd_{}'.format(size)
            if data_format == 'nhwc':
                nch = s[2]
                X = tf.transpose(X, perm=[0, 3, 1, 2])
                X = tf.reshape(X, [-1, s[0], s[1]])
            else:
                nch = s[0]
                X = tf.reshape(X, [-1, s[1], s[2]])

        # Do the dtcwt, now with a 3 dimensional input
        with tf.variable_scope(name):
            Yl, Yh, Yscale = self._forward_ops(X, nlevels)

        # Put the channels back into their correct positions
        with tf.variable_scope('batch_to_ch'):
            # Reshape Yl
            s = Yl.get_shape().as_list()[1:]
            Yl = tf.reshape(Yl, [-1, nch, s[0], s[1]], name='Yl_reshape')
            if data_format == 'nhwc':
                Yl = tf.transpose(Yl, [0, 2, 3, 1], name='Yl_ch_to_end')

            # Reshape Yh
            with tf.variable_scope('Yh'):
                Yh_new = [None,] * nlevels
                for i in range(nlevels):
                    s = Yh[i].get_shape().as_list()[1:]
                    Yh_new[i] = tf.reshape(
                        Yh[i], [-1, nch, s[0], s[1], s[2]],
                        name='scale{}_reshape'.format(i))
                    if data_format == 'nhwc':
                        Yh_new[i] = tf.transpose(
                            Yh_new[i], [0, 2, 3, 1, 4],
                            name='scale{}_ch_to_end'.format(i))
                Yh = tuple(Yh_new)

            # Reshape Yscale
            if include_scale:
                with tf.variable_scope('Yscale'):
                    Yscale_new = [None,] * nlevels
                    for i in range(nlevels):
                        s = Yscale[i].get_shape().as_list()[1:]
                        Yscale_new[i] = tf.reshape(
                            Yscale[i], [-1, nch, s[0], s[1]],
                            name='scale{}_reshape'.format(i))
                        if data_format == 'nhwc':
                            Yscale_new[i] = tf.transpose(
                                Yscale_new[i], [0, 2, 3, 1],
                                name='scale{}_ch_to_end'.format(i))
                    Yscale = tuple(Yscale_new)

            if include_scale:
                return Pyramid(Yl, Yh, Yscale, numpy)
            else:
                return Pyramid(Yl, Yh, None, numpy)

    def inverse(self, pyramid, gain_mask=None):
        """ Perform an inverse transform on an image.

        Can provide the inverse transform with either an np array (naive
        usage), or a tensorflow variable or placeholder (designed usage).

        :param pyramid: A :py:class:`dtcwt.tf.Pyramid` like class holding
            the transform domain representation to invert
        :param gain_mask: Gain to be applied to each sub-band. Should have shape
            (6, nlevels) or be None.

        :returns: Either a tf.Variable or a numpy array compatible with the
            reconstruction.

        .. note::

            A tf.Variable is returned if the pyramid input was a Pyramid class.
            If it wasn't, then, we return a numpy array (note that this is
            inefficient, as in both cases we have to construct the graph - in
            the second case, we then execute it and discard it).

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with
        direction *d* at level *l*. If gain_mask[d,l] == 0, no computation is
        performed for band (d,l). Default *gain_mask* is all ones. Note that
        both *d* and *l* are zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
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
        extended = False
        Yl_shape = tuple(Yl.get_shape().as_list())
        if len(Yl_shape) == 2:
            Yl = tf.expand_dims(Yl, axis=0)
            Yh = tuple(tf.expand_dims(x, axis=0) for x in Yh)
            extended = True
        elif len(Yl_shape) == 4:
            raise ValueError(
                'The entered variables have too many ' +
                'dimensions - ' + str(Yl_shape) + '. For batches of ' +
                'images with multiple channels (i.e. 4 dimensions), ' +
                'please either enter each channel separately, or use ' +
                'the inverse_channels method.')

        # Do the inverse transform
        s = Yl.get_shape().as_list()[1:]
        nlevels = len(Yh)
        size = '{}x{}_up_{}'.format(s[0], s[1], nlevels)
        name = 'dtcwt_inv_{}'.format(size)
        with tf.name_scope(name):
            X = self._inverse_ops(Yl, Yh, gain_mask)

        # Return data in a shape the user was expecting
        if extended:
            X = X[0]

        if numpy:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                X = sess.run(X)

        return X

    def inverse_channels(self, pyramid, gain_mask=None, data_format="nhwc"):
        """
        Perform an inverse transform on an image with multiple channels.

        Must provide with a tensorflow variable or placeholder (unlike the more
        general :py:method:`Transform2d.inverse`).

        :param pyramid: A :py:class:`dtcwt.tf.Pyramid` like class holding
            the transform domain representation to invert
        :param gain_mask: Gain to be applied to each subband. Should have shape
            [6, nlevels].
        :param data_format: An optional string of the form "nchw" or "nhwc",
            specifying the data format of the input. If format is "nchw" (the
            default), then data are in the form [batch, channels, h, w] for Yl
            and [batch, channels, h, w, 6] for Yh. If the format is "nhwc", then
            the data are in the form [batch, h, w, c] for Yl and
            [batch, h, w, c, 6] for Yh.

        :returns: A tf.Variable, X, compatible with the reconstruction.

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with
        direction *d* at level *l*. If gain_mask[d,l] == 0, no computation is
        performed for band (d,l). Default *gain_mask* is all ones. Note that
        both *d* and *l* are zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
        """
        # Input checking
        data_format = data_format.lower()
        if data_format != "nchw" and data_format != "nhwc":
            raise ValueError('The data format must be either "ncwh" or ' +
                             '"nhwc", not {}'.format(data_format))
        if data_format == "nhwc":
            channel_ax = 3
        else:
            channel_ax = 1

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

        # Check the shape was 4D
        Yl_shape = Yl.get_shape().as_list()
        if len(Yl_shape) != 4:
            raise ValueError(
                """The entered lowpass variable has incorrect dimensions {}.
                for data_format of {}.""".format(Yl_shape, data_format))

        # Move all of the channels into the batch dimension for the lowpass
        # input. This may involve transposing, depending on the data format
        with tf.variable_scope('ch_to_batch'):
            s = Yl.get_shape().as_list()
            num_channels = s[channel_ax]
            nlevels = len(Yh)
            if data_format == "nhwc":
                size = '{}x{}_up_{}'.format(s[1], s[2], nlevels)
                Yl_new = tf.transpose(Yl, [0, 3, 1, 2])
                Yl_new = tf.reshape(Yl_new, [-1, s[1], s[2]])
            else:
                size = '{}x{}_up_{}'.format(s[2], s[3], nlevels)
                Yl_new = tf.reshape(Yl, [-1, s[2], s[3]])

            # Move all of the channels into the batch dimension for the highpass
            # input. This may involve transposing, depending on the data format
            Yh_new = []
            for scale in Yh:
                s = scale.get_shape().as_list()
                if s[channel_ax] != num_channels:
                    raise ValueError(
                        """The number of channels has to be consistent for all
                        inputs across the channel axis {}. You fed in Yl: {}
                        and Yh: {}""".format(channel_ax, Yl, Yh))
                if data_format == "nhwc":
                    scale = tf.transpose(scale, [0, 3, 1, 2, 4])
                    Yh_new.append(tf.reshape(scale, [-1, s[1], s[2], s[4]]))
                else:
                    Yh_new.append(tf.reshape(scale, [-1, s[2], s[3], s[4]]))

        name = 'dtcwt_inv_{}_{}channels'.format(size, num_channels)
        with tf.variable_scope(name):
            X = self._inverse_ops(Yl_new, Yh_new, gain_mask)

        with tf.variable_scope('batch_to_ch'):
            s = X.get_shape().as_list()
            X = tf.reshape(X, [-1, num_channels, s[1], s[2]])
            if data_format == "nhwc":
                X = tf.transpose(X, [0, 2, 3, 1], name='X')

        if numpy:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                X = sess.run(X)

        return X

    def _forward_ops(self, X, nlevels=3):
        """ Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 3D real array of size [batch, h, w]
        :param nlevels: Number of levels of wavelet decomposition
        :param extended: True if a singleton dimension was added at the
            beginning of the input. Signal to remove afterwards.

        :returns: A tuple of Yl, Yh, Yscale
        """

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift[:10]
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        # Check the shape and form of the input
        if X.dtype not in tf_dtypes:
            raise ValueError('X needs to be a tf variable or placeholder')

        original_size = X.get_shape().as_list()[1:]

        if len(original_size) >= 3:
            raise ValueError(
                """The entered variable has too many dimensions {}. If
                the final dimension are colour channels, please enter each
                channel separately.""".format(original_size))

        # ############################ Resize #################################
        # The next few lines of code check to see if the image is odd in size,
        # if so an extra ... row/column will be added to the bottom/right of the
        # image
        initial_row_extend = 0
        initial_col_extend = 0
        # If the row count of X is not divisible by 2 then we need to
        # extend X by adding a row at the bottom
        if original_size[0] % 2 != 0:
            bottom_row = tf.slice(X, [0, original_size[0] - 1, 0], [-1, 1, -1])
            X = tf.concat([X, bottom_row], axis=1)
            initial_row_extend = 1

        # If the col count of X is not divisible by 2 then we need to
        # extend X by adding a col to the right
        if original_size[1] % 2 != 0:
            right_col = tf.slice(X, [0, 0, original_size[1] - 1], [-1, -1, 1])
            X = tf.concat([X, right_col], axis=2)
            initial_col_extend = 1

        extended_size = X.get_shape().as_list()[1:3]

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
            Lo = colfilter(X, h0o)
            Hi = colfilter(X, h1o)
            if len(self.biort) >= 6:
                Ba = colfilter(X, h2o)

            # Do odd top-level filters on rows.
            LoLo = rowfilter(Lo, h0o)
            LoLo_shape = LoLo.get_shape().as_list()[1:]

            # Horizontal wavelet pair (15 & 165 degrees)
            horiz = q2c(rowfilter(Hi, h0o))

            # Vertical wavelet pair (75 & 105 degrees)
            vertic = q2c(rowfilter(Lo, h1o))

            # Diagonal wavelet pair (45 & 135 degrees)
            if len(self.biort) >= 6:
                diag = q2c(rowfilter(Ba, h2o))
            else:
                diag = q2c(rowfilter(Hi, h1o))

            # Pack all 6 tensors into one
            Yh[0] = tf.stack(
                [horiz[0], diag[0], vertic[0], vertic[1], diag[1], horiz[1]],
                axis=3)

            Yscale[0] = LoLo

        # ############################ Level 2+ ################################
        # Uses the qshift filters
        for level in xrange(1, nlevels):
            row_size, col_size = LoLo_shape[0], LoLo_shape[1]
            # If the row count of LoLo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra rows to make it so
            if row_size % 4 != 0:
                LoLo = tf.pad(LoLo, [[0, 0], [1, 1], [0, 0]], 'SYMMETRIC')

            # If the col count of LoLo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra cols to make it so
            if col_size % 4 != 0:
                LoLo = tf.pad(LoLo, [[0, 0], [0, 0], [1, 1]], 'SYMMETRIC')

            # Do even Qshift filters on cols.
            Lo = coldfilt(LoLo, h0b, h0a)
            Hi = coldfilt(LoLo, h1b, h1a)
            if len(self.qshift) >= 12:
                Ba = coldfilt(LoLo, h2b, h2a)

            # Do even Qshift filters on rows.
            LoLo = rowdfilt(Lo, h0b, h0a)
            LoLo_shape = LoLo.get_shape().as_list()[1:3]

            # Horizontal wavelet pair (15 & 165 degrees)
            horiz = q2c(rowdfilt(Hi, h0b, h0a))

            # Vertical wavelet pair (75 & 105 degrees)
            vertic = q2c(rowdfilt(Lo, h1b, h1a))

            # Diagonal wavelet pair (45 & 135 degrees)
            if len(self.qshift) >= 12:
                diag = q2c(rowdfilt(Ba, h2b, h2a))
            else:
                diag = q2c(rowdfilt(Hi, h1b, h1a))

            # Pack all 6 tensors into one
            Yh[level] = tf.stack(
                [horiz[0], diag[0], vertic[0], vertic[1], diag[1], horiz[1]],
                axis=3)

            Yscale[level] = LoLo

        Yl = LoLo

        if initial_row_extend == 1 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                """The bottom row and rightmost column have been duplicated,
                prior to decomposition.""")

        if initial_row_extend == 1 and initial_col_extend == 0:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The bottom row has been duplicated, prior to decomposition.')

        if initial_row_extend == 0 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                """The rightmost column has been duplicated, prior to
                decomposition.""")

        return Yl, tuple(Yh), tuple(Yscale)

    def _inverse_ops(self, Yl, Yh, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
        reconstruction.

        :param Yl: The lowpass output from a forward transform. Should be a
            tensorflow variable.
        :param Yh: The tuple of highpass outputs from a forward transform.
            Should be tensorflow variables.
        :param gain_mask: Gain to be applied to each subband.

        :returns: A tf.Variable holding the output

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with
        direction *d* at level *l*. If gain_mask[d,l] == 0, no computation is
        performed for band (d,l). Default *gain_mask* is all ones. Note that
        both *d* and *l* are zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        a = len(Yh)  # No of levels.

        if gain_mask is None:
            gain_mask = np.ones((6, a))  # Default gain_mask.

        gain_mask = np.array(gain_mask)

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, \
                g1a, g1b, h2a, h2b, g2a, g2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        current_level = a
        Z = Yl

        # This ensures that for level 1 we never do the following
        while current_level >= 2:
            lh = c2q(Yh[current_level - 1][:, :, :, 0:6:5],
                     gain_mask[[0, 5],
                     current_level - 1])
            hl = c2q(Yh[current_level - 1][:, :, :, 2:4:1],
                     gain_mask[[2, 3],
                     current_level - 1])
            hh = c2q(Yh[current_level - 1][:, :, :, 1:5:3],
                     gain_mask[[1, 4],
                     current_level - 1])

            # Do even Qshift filters on columns.
            y1 = colifilt(Z, g0b, g0a) + colifilt(lh, g1b, g1a)

            if len(self.qshift) >= 12:
                y2 = colifilt(hl, g0b, g0a)
                y2bp = colifilt(hh, g2b, g2a)

                # Do even Qshift filters on rows.
                y1T = tf.transpose(y1, perm=[0, 2, 1])
                y2T = tf.transpose(y2, perm=[0, 2, 1])
                y2bpT = tf.transpose(y2bp, perm=[0, 2, 1])
                Z = tf.transpose(
                    colifilt(y1T, g0b, g0a) +
                    colifilt(y2T, g1b, g1a) +
                    colifilt(y2bpT, g2b, g2a),
                    perm=[0, 2, 1])
            else:
                y2 = colifilt(hl, g0b, g0a) + colifilt(hh, g1b, g1a)

                # Do even Qshift filters on rows.
                y1T = tf.transpose(y1, perm=[0, 2, 1])
                y2T = tf.transpose(y2, perm=[0, 2, 1])
                Z = tf.transpose(
                    colifilt(y1T, g0b, g0a) +
                    colifilt(y2T, g1b, g1a),
                    perm=[0, 2, 1])

            # Check size of Z and crop as required
            Z_r, Z_c = Z.get_shape().as_list()[1:3]
            S_r, S_c = Yh[current_level - 2].get_shape().as_list()[1:3]
            # check to see if this result needs to be cropped for the rows
            if Z_r != S_r * 2:
                Z = Z[:, 1:-1, :]
            # check to see if this result needs to be cropped for the cols
            if Z_c != S_c * 2:
                Z = Z[:, :, 1:-1]

            # Assert that the size matches at this stage
            Z_r, Z_c = Z.get_shape().as_list()[1:3]
            if Z_r != S_r * 2 or Z_c != S_c * 2:
                raise ValueError(
                    'Sizes of highpasses {}x{} are not '.format(Z_r, Z_c) +
                    'compatible with {}x{} from next level'.format(S_r, S_c))

            current_level = current_level - 1

        if current_level == 1:
            lh = c2q(Yh[current_level - 1][:, :, :, 0:6:5],
                     gain_mask[[0, 5],
                     current_level - 1])
            hl = c2q(Yh[current_level - 1][:, :, :, 2:4:1],
                     gain_mask[[2, 3],
                     current_level - 1])
            hh = c2q(Yh[current_level - 1][:, :, :, 1:5:3],
                     gain_mask[[1, 4],
                     current_level - 1])

            # Do odd top-level filters on columns.
            y1 = colfilter(Z, g0o) + colfilter(lh, g1o)

            if len(self.biort) >= 6:
                y2 = colfilter(hl, g0o)
                y2bp = colfilter(hh, g2o)

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o) + rowfilter(y2, g1o) + \
                    rowfilter(y2bp, g2o)
            else:
                y2 = colfilter(hl, g0o) + colfilter(hh, g1o)

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o) + rowfilter(y2, g1o)

        return Z


def q2c(y):
    """
    Convert from quads in y to complex numbers in z.
    """

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d
    # Combine (a,b) and (d,c) to form two complex subimages.
    a, b = y[:, 0::2, 0::2], y[:, 0::2, 1::2]
    c, d = y[:, 1::2, 0::2], y[:, 1::2, 1::2]

    p = tf.complex(a / np.sqrt(2), b / np.sqrt(2))    # p = (a + jb) / sqrt(2)
    q = tf.complex(d / np.sqrt(2), -c / np.sqrt(2))   # q = (d - jc) / sqrt(2)

    # Form the 2 highpasses in z.
    return (p - q, p + q)


def c2q(w, gain):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """

    # Input has shape [batch, r, c, 2]
    r, c = w.get_shape().as_list()[1:3]

    sc = np.sqrt(0.5) * gain
    P = w[:, :, :, 0] * sc[0] + w[:, :, :, 1] * sc[1]
    Q = w[:, :, :, 0] * sc[0] - w[:, :, :, 1] * sc[1]

    # Recover each of the 4 corners of the quads.
    x1 = tf.real(P)
    x2 = tf.imag(P)
    x3 = tf.imag(Q)
    x4 = -tf.real(Q)

    # Stack 2 inputs of shape [batch, r, c] to [batch, r, 2, c]
    x_rows1 = tf.stack([x1, x3], axis=-2)
    # Reshaping interleaves the results
    x_rows1 = tf.reshape(x_rows1, [-1, 2 * r, c])
    # Do the same for the even columns
    x_rows2 = tf.stack([x2, x4], axis=-2)
    x_rows2 = tf.reshape(x_rows2, [-1, 2 * r, c])

    # Stack the two [batch, 2*r, c] tensors to [batch, 2*r, c, 2]
    x_cols = tf.stack([x_rows1, x_rows2], axis=-1)
    y = tf.reshape(x_cols, [-1, 2 * r, 2 * c])

    return y
