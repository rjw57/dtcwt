from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import logging

from six.moves import xrange

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import asfarray
from dtcwt.numpy import Transform2d as Transform2dNumPy
from dtcwt.numpy import Pyramid as Pyramid_np
from dtcwt.tf import Pyramid_tf

from dtcwt.tf.lowlevel import *

def dtwavexfm2(X, nlevels=3, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, include_scale=False):
    t = Transform2d(biort=biort, qshift=qshift)
    r = t.forward(X, nlevels=nlevels, include_scale=include_scale)
    if include_scale:
        return r.lowpass, r.highpasses, r.scales
    else:
        return r.lowpass, r.highpasses

def dtwaveifm2(Yl, Yh, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT, gain_mask=None):
    t = Transform2d(biort=biort, qshift=qshift)
    r = t.inverse(Pyramid_np(Yl, Yh), gain_mask=gain_mask)
    return r


class Transform2d(Transform2dNumPy):
    """
    An implementation of the 2D DT-CWT via Tensorflow. 
    *biort* and *qshift* are the wavelets which parameterise the transform.
    If *biort* or *qshift* are strings, they are used as an argument to the
    :py:func:`dtcwt.coeffs.biort` or :py:func:`dtcwt.coeffs.qshift` functions.
    Otherwise, they are interpreted as tuples of vectors giving filter
    coefficients. In the *biort* case, this should be (h0o, g0o, h1o, g1o). In
    the *qshift* case, this should be (h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b).
    
    Creating an object of this class loads the necessary filters onto the 
    tensorflow graph. A subsequent call to :py:func:`Transform2d.forward` with 
    a placeholder will create a forward transform for an input of the placeholder's
    size. You can evaluate the resulting ops several times feeding different
    images into the placeholder *assuming* they have the same resolution. For 
    a different resolution image, call the :py:func:`Transform2d.forward` 
    function again.
    """

    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        super(Transform2d, self).__init__(biort=biort, qshift=qshift)
        # Use our own graph when the user calls forward with numpy arrays
        self.np_graph = tf.Graph()
        self.forward_graphs = {}
        self.inverse_graphs = {}

    def _find_forward_graph(self, shape):
        ''' See if we can reuse an old graph for the forward transform '''
        find_key = '{}x{}'.format(shape[0], shape[1])
        for key,val in self.forward_graphs.items():
            if find_key == key:
                return val
        return None

    def _add_forward_graph(self, p_ops, shape):
        ''' Keep record of the pyramid so we can use it later if need be '''
        find_key = '{}x{}'.format(shape[0], shape[1])
        self.forward_graphs[find_key] = p_ops

    def _find_inverse_graph(self, Lo_shape, nlevels):
        ''' See if we can reuse an old graph for the inverse transform '''
        find_key = '{}x{}'.format(Lo_shape[0], Lo_shape[1])
        for key,val in self.forward_graphs.items():
            if find_key == key:
                return val
        return None

    def _add_inverse_graph(self, p_ops, Lo_shape, nlevels):
        ''' Keep record of the pyramid so we can use it later if need be '''
        find_key = '{}x{} up {}'.format(Lo_shape[0], Lo_shape[1], nlevels)
        self.inverse_graphs[find_key] = p_ops

    def forward(self, X, nlevels=3, include_scale=False):
        '''
        Perform a forward transform on an image. Can provide the forward
        transform with either an np array (naive usage), or a tensorflow
        variable or placeholder (designed usage).
        If a numpy array is provided, the forward function will create a graph
        of the right size to match the input (or check if it has previously
        created one), and then feed the input into the graph and evaluate it.
        This operation will return a Pyramid() object similar to running the
        numpy version would.
        If a tensorflow variable or placeholder is provided, the forward
        function will create a graph of the right size, and return
        a Pyramid_ops() object.
        '''

        # Check if a numpy array was provided
        if not isinstance(X, tf.Tensor) and not isinstance(X, tf.Variable):
            X = np.atleast_2d(asfarray(X))
            if len(X.shape) >= 3: 
                raise ValueError('''The entered variable has incorrect dimensions {}.
                    If X is a numpy array (or any non tensorflow object), it
                    must be of shape [height, width]. For colour images, please
                    enter each channel separately. If you wish to enter a batch
                    of images, please instead provide either a tf.Placeholder
                    or a tf.Variable input of size [batch, height, width].
                    '''.format(original_size))         

            # Check if the ops already exist for an input of the given size
            p_ops = self._find_forward_graph(X.shape)

            # If not, create a graph
            if p_ops is None:
                ph = tf.placeholder(tf.float32, [None, X.shape[0], X.shape[1]])
                size = '{}x{}'.format(X.shape[0], X.shape[1])
                name = 'dtcwt_fwd_{}'.format(size)
                with self.np_graph.name_scope(name):
                    p_ops = self._forward_ops(ph, nlevels, include_scale)

                self._add_forward_graph(p_ops, X.shape)

            # Evaluate the graph with the given input
            with self.np_graph.as_default():
                return p_ops.eval_fwd(X)

        # A tensorflow object was provided
        else: 
            X_shape = X.get_shape().as_list()
            if len(X_shape) != 3: 
                raise ValueError('''The entered variable has incorrect dimensions {}.
                    If X is a tf placeholder or variable, it must be of shape
                    [batch, height, width] (batch can be None). For colour images, 
                    please enter each channel separately. 
                    '''.format(original_size))         

            original_size = X.get_shape().as_list()[1:]
            size = '{}x{}'.format(original_size[0], original_size[1])
            name = 'dtcwt_fwd_{}'.format(size)
            with tf.name_scope(name):
                return self._forward_ops(X, nlevels, include_scale)

    def inverse(self, pyramid, gain_mask=None):
        '''
        Perform an inverse transform on an image. Can provide the inverse 
        transform with either an np array (naive usage), or a tensorflow
        variable or placeholder (designed usage).

        :param pyramid: A :py:class:`dtcwt.Pyramid` or
        `:py:class:`dtcwt.tf.Pyramid_tf` like class holding the transform
        domain representation to invert
        :param gain_mask: Gain to be applied to each subband. Should have shape
        [6, nlevels]. 
        :returns: Either a tf.Variable or a numpy array compatible with the
        reconstruction. A tf.Variable is returned if the pyramid input was
        a Pyramid_tf class. If it wasn't, then, we return a numpy array (note
        that this is inefficient, as in both cases we have to construct the
        graph - in the second case, we then execute it and discard it).

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with direction
        *d* at level *l*. If gain_mask[d,l] == 0, no computation is performed for
        band (d,l). Default *gain_mask* is all ones. Note that both *d* and *l* are
        zero-indexed.

        '''

        # Check if a numpy array was provided
        if isinstance(pyramid, Pyramid_np) or (hasattr(pyramid, 'lowpass') 
                and hasattr(pyramid, 'highpasses')):

            Yl, Yh = pyramid.lowpass, pyramid.highpasses

            # Check if the ops already exist for an input of the given size
            nlevels = len(Yh)
            p_ops = self._find_inverse_graph(Yl.shape, nlevels)

            # If not, create a graph
            if p_ops is None:
                Lo_ph = tf.placeholder(tf.float32, [None, Yl.shape[0], Yl.shape[1]])
                Hi_ph = tuple(
                            tf.placeholder(tf.complex64, [None, *level.shape]) for
                            level in Yh)
                p_in = Pyramid_tf(None, Lo_ph, Hi_ph)
                size = '{}x{}_up_{}'.format(Yl.shape[0], Yl.shape[1], nlevels)
                name = 'dtcwt_inv_{}'.format(size) 

                with self.np_graph.name_scope(name):
                    p_ops = self._inverse_ops(p_in, gain_mask)

                # keep record of the pyramid so we can use it later if need be
                self.forward_graphs[size] = p_ops

            # Evaluate the graph with the given input
            with self.np_graph.as_default():
                return p_ops.eval_inv(Yl, Yh)

        # A tensorflow object was provided
        elif isinstance(pyramid, Pyramid_tf): 
            s = pyramid.lowpass_op.get_shape().as_list()[1:]
            nlevels = len(pyramid.highpasses_ops)
            size = '{}x{}_up_{}'.format(s[0], s[1], nlevels)
            name = 'dtcwt_inv_{}'.format(size)
            with tf.name_scope(name):
                return self._inverse_ops(pyramid, gain_mask)
        else:
            raise ValueError('''Unknown pyramid provided to inverse transform''')
    

    def _forward_ops(self, X, nlevels=3, include_scale=False):
        """Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.
        :param X: 3D real array of size [Batch, rows, cols]
        :param nlevels: Number of levels of wavelet decomposition
        :param include_scale: True if you want to receive the lowpass coefficients at
            intermediate layers.
        :returns: A :py:class:`dtcwt.Pyramid` compatible object representing the transform-domain signal
        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
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
        if not isinstance(X, tf.Tensor) and not isinstance(X, tf.Variable):
            raise ValueError('''Please provide the forward function with 
                a tensorflow placeholder or variable of size [batch, width,
                height] (batch can be None if you do not wish to specify it).''')

        original_size = X.get_shape().as_list()[1:]

        if len(original_size) >= 3:
            raise ValueError('The entered variable has too many dimensions {}. If '
                    'the final dimension are colour channels, please enter each ' +
                    'channel separately.'.format(original_size))

        # Save the input placeholder/variable
        X_in = X
        ############################## Resize #################################
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
            if include_scale:
                return Pyramid_tf(X_in, X, (), ())
            else:
                return Pyramid_tf(X_in, X, ())

        
        ############################ Initialise ###############################
        Yh = [None,] * nlevels
        if include_scale:
            # this is only required if the user specifies a third output component.
            Yscale = [None,] * nlevels

        ############################# Level 1 #################################
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
            
            if include_scale:
                Yscale[0] = LoLo
                
                
        ############################# Level 2+ ################################
        # Uses the qshift filters        
        for level in xrange(1, nlevels):
            row_size, col_size = LoLo_shape[0], LoLo_shape[1]
            # If the row count of LoLo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra rows to make it so
            if row_size % 4 != 0:
                LoLo = tf.pad(LoLo, [[0, 0], [1, 1], [0, 0]], 'SYMMETRIC')    
                #top_row = tf.slice(LoLo, [0, 0, 0], [-1, 1, -1])
                #bottom_row = tf.slice(LoLo, [0, row_size - 1, 0], [-1, 1, -1])
                #LoLo = tf.concat([top_row, LoLoLoLo, bottom_row], axis=1)

            # If the col count of LoLo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra cols to make it so
            if col_size % 4 != 0:
                LoLo = tf.pad(LoLo, [[0, 0], [0, 0], [1, 1]], 'SYMMETRIC')    
                #right_col = tf.slice(LoLo, [0, 0, col_size - 2], [-1, -1, 2])
                #LoLo = tf.concat([LoLo, right_col], axis=2)

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
            
            if include_scale:
                Yscale[level] = LoLo
            
        Yl = LoLo
        
        if initial_row_extend == 1 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in original_size))))
            logging.warn(
                'The bottom row and rightmost column have been duplicated, prior to decomposition.')

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
                'The rightmost column has been duplicated, prior to decomposition.')

        if include_scale:
            return Pyramid_tf(X_in, Yl, tuple(Yh), tuple(Yscale))
        else:
            return Pyramid_tf(X_in, Yl, tuple(Yh))
    

    def _inverse_ops(self, pyramid, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
        reconstruction.

        :param pyramid: A :py:class:`dtcwt.tf.Pyramid_tf`-like class holding the 
        transform domain representation to invert.
        :param gain_mask: Gain to be applied to each subband.

        :returns: A :py:class:`dtcwt.tf.Pyramid_tf` class which can be
        evaluated to get the inverted signal, X.

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with direction
        *d* at level *l*. If gain_mask[d,l] == 0, no computation is performed for
        band (d,l). Default *gain_mask* is all ones. Note that both *d* and *l* are
        zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        Yl = pyramid.lowpass_op
        Yh = pyramid.highpasses_ops

        a = len(Yh) # No of levels.

        if gain_mask is None:
            gain_mask = np.ones((6, a)) # Default gain_mask.

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
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b, g2a, g2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        current_level = a
        Z = Yl

        while current_level >= 2: # this ensures that for level 1 we never do the following
            lh = c2q(Yh[current_level-1][:,:,:,0:6:5], gain_mask[[0, 5], current_level-1])
            hl = c2q(Yh[current_level-1][:,:,:,2:4:1], gain_mask[[2, 3], current_level-1])
            hh = c2q(Yh[current_level-1][:,:,:,1:5:3], gain_mask[[1, 4], current_level-1])

            # Do even Qshift filters on columns.
            y1 = colifilt(Z, g0b, g0a) + colifilt(lh, g1b, g1a)

            if len(self.qshift) >= 12:
                y2 = colifilt(hl, g0b, g0a)
                y2bp = colifilt(hh, g2b, g2a)

                # Do even Qshift filters on rows.
                Z = tf.transpose(colifilt(tf.transpose(y1,perm=[0,2,1]), g0b, g0a) + 
                                 colifilt(tf.transpose(y2,perm=[0,2,1]), g1b, g1a) + 
                                 colifilt(tf.transpose(y2bp,perm=[0,2,1]), g2b, g2a),
                        perm=[0,2,1])
            else:
                y2 = colifilt(hl, g0b, g0a) + colifilt(hh, g1b, g1a)

                # Do even Qshift filters on rows.
                Z = tf.transpose(colifilt(tf.transpose(y1, perm=[0,2,1]), g0b, g0a) + 
                                 colifilt(tf.transpose(y2, perm=[0,2,1]), g1b, g1a),
                        perm=[0,2,1])

            # Check size of Z and crop as required
            Z_r, Z_c = Z.get_shape().as_list()[1:3]
            S_r, S_c = Yh[current_level-2].get_shape().as_list()[1:3]
            # check to see if this result needs to be cropped for the rows
            if Z_r != S_r * 2:    
                Z = Z[:,1:-1,:]
            # check to see if this result needs to be cropped for the cols
            if Z_c != S_c*2:    
                Z = Z[:,:,1:-1]

            # Assert that the size matches at this stage
            Z_r, Z_c = Z.get_shape().as_list()[1:3]
            if Z_r != S_r * 2 or Z_c != S_c*2: 
                raise ValueError('Sizes of highpasses are not valid for DTWAVEIFM2')
            
            current_level = current_level - 1

        if current_level == 1:
            lh = c2q(Yh[current_level-1][:,:,:,0:6:5],gain_mask[[0, 5],current_level-1])
            hl = c2q(Yh[current_level-1][:,:,:,2:4:1],gain_mask[[2, 3],current_level-1])
            hh = c2q(Yh[current_level-1][:,:,:,1:5:3],gain_mask[[1, 4],current_level-1])

            # Do odd top-level filters on columns.
            y1 = colfilter(Z, g0o) + colfilter(lh, g1o)

            if len(self.biort) >= 6:
                y2 = colfilter(hl, g0o)
                y2bp = colfilter(hh, g2o)

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o) + rowfilter(y2, g1o) + rowfilter(y2bp, g2o)
            else:
                y2 = colfilter(hl, g0o) + colfilter(hh, g1o)

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o) + rowfilter(y2, g1o)

        return Pyramid_tf(Z, Yl, Yh)

        

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
    a,b,c,d = y[:, 0::2, 0::2], y[:, 0::2,1::2], y[:, 1::2,0::2], y[:, 1::2,1::2]
    
    p = tf.complex(a/np.sqrt(2), b/np.sqrt(2))    # p = (a + jb) / sqrt(2)
    q = tf.complex(d/np.sqrt(2), -c/np.sqrt(2))   # q = (d - jc) / sqrt(2)

    # Form the 2 highpasses in z.
    return (p-q, p+q)        


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
    r,c = w.get_shape().as_list()[1:3]

    sc = np.sqrt(0.5) * gain
    P = w[:,:,:,0]*sc[0] + w[:,:,:,1]*sc[1]
    Q = w[:,:,:,0]*sc[0] - w[:,:,:,1]*sc[1]

    # Recover each of the 4 corners of the quads.
    x1 = tf.real(P)
    x2 = tf.imag(P)
    x3 = tf.real(Q)
    x4 = -tf.imag(Q)

    # Stack 2 inputs of shape [batch, r, c] to [batch, r, 2, c]
    x_rows1 = tf.stack([x1,x3], axis=2)
    # Reshaping interleaves the results
    x_rows1 = tf.reshape(x_rows1, [-1, 2*r, c])
    # Do the same for the even columns
    x_rows2 = tf.stack([x2,x3], axis=2)
    x_rows2 = tf.reshape(x_rows2, [-1, 2*r, c])

    # Stack the two [batch, 2*r, c] tensors to [batch, 2*r, c, 2]
    x_cols = tf.stack([x_rows1, x_rows2], axis=-1)
    y = tf.reshape(x_cols, [-1, 2*r, 2*c])

    return y

