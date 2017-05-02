from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import logging

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT
from dtcwt.utils import asfarray

from dtcwt.numpy import Pyramid as Pyramid_np

class Pyramid_tf(object):
    """A tensorflow representation of a transform domain signal.
    Backends are free to implement any class which respects this interface for
    storing transform-domain signals, so long as the attributes have the
    correct names and are tensorflow tensors (or placeholders).
    The inverse transform may accept a backend-specific version of this class
    but should always accept any class which corresponds to this interface.

    .. py:attribute:: X
        A placeholder which the user can use when they want to evaluate the
        forward dtcwt. 
    .. py:attribute:: lowpass_op
        A tensorflow tensor that can be evaluated in a session to return
        the coarsest scale lowpass signal for the input, X. 
    .. py:attribute:: highpasses_op
        A tuple of tensorflow tensors, where each element is the complex
        subband coefficients for corresponding scales finest to coarsest.
    .. py:attribute:: scales
        *(optional)* A tuple where each element is a tensorflow tensor 
        containing the lowpass signal for corresponding scales finest to
        coarsest. This is not required for the inverse and may be *None*.
    .. py:method:: apply_reshaping(fn)
        A helper method to apply a tensor reshaping to all of the elements in
        the pyramid.
    .. py:method:: eval_fwd(X)
        A helper method to evaluate the forward transform, feeding *X* as input
        to the tensorflow session. Assumes that the object was returned from
        the Transform2d().forward() method.
    .. py:method:: eval_inv(Yl, Yh)
        A helper method to evaluate the inverse transform, feeding *Yl* and
        *Yh* to the tensorflow session. Assumes that the object was returned
        from the Trasnform2d().inverse() method.
    """
    def __init__(self, X, lowpass, highpasses, scales=None,
                 graph=tf.get_default_graph()):
        self.X = X
        self.lowpass_op = lowpass
        self.highpasses_ops = highpasses
        self.scales_ops = scales
        self.graph = graph

    def _get_lowpass(self, data, sess=None):
        if self.lowpass_op is None:
            return None

        if sess is None:
            sess = tf.Session(graph=self.graph)

        with sess:
            try:
                y = sess.run(self.lowpass_op, {self.X : data})
            except ValueError:
                y = sess.run(self.lowpass_op, {self.X : [data]})[0]
        return y
        
    def _get_highpasses(self, data, sess=None):
        if self.highpasses_ops is None:
            return None

        if sess is None:
            sess = tf.Session(graph=self.graph)

        with sess:
            try: 
                y = tuple(
                        [sess.run(layer_hp, {self.X : data}) 
                        for layer_hp in self.highpasses_ops])
            except ValueError:
                y = tuple(
                        [sess.run(layer_hp, {self.X : [data]})[0] 
                        for layer_hp in self.highpasses_ops])
        return y

    def _get_scales(self, data, sess=None):
        if self.scales_ops is None:
            return None

        if sess is None:
            sess = tf.Session(graph=self.graph)

        with sess:
            try:
                y = tuple(
                        sess.run(layer_scale, {self.X : data})
                        for layer_scale in self.scales_ops)
            except ValueError:
                y = tuple(
                        sess.run(layer_scale, {self.X : [data]})[0]
                        for layer_scale in self.scales_ops)
        return y

    def _get_X(self, Yl, Yh, sess=None):
        if self.X is None:
            return None

        if sess is None:
            sess = tf.Session(graph=self.graph)

        with sess:
            try:
                # Use dictionary comprehension to feed in our Yl and our
                # multiple layers of Yh
                data = [Yl, *list(Yh)]
                placeholders = [self.lowpass_op, *list(self.highpasses_ops)]
                X = sess.run(self.X, {i : d for i,d in zip(placeholders,data)})
            except ValueError:
                data = [Yl, *list(Yh)]
                placeholders = [self.lowpass_op, *list(self.highpasses_ops)]
                X = sess.run(self.X, {i : [d] for i,d in zip(placeholders,data)})[0]
        return X


    def apply_reshaping(self, fn):
        """
        A helper function to apply a tensor transformation on all of the
        elements in the pyramid. E.g. reshape all of them in the same way.

        :param fn: function to apply to each of the lowpass_op, highpasses_ops and
            scale_ops tensors
        """
        self.lowpass_op = fn(self.lowpass_op)
        self.highpasses_ops = tuple(
                    [fn(h_scale) for h_scale in self.highpasses_ops])
        if not self.scales_ops is None:
            self.scales_ops = tuple(
                    [fn(s_scale) for s_scale in self.scales_ops])


    def eval_fwd(self, X, sess=None):
        """
        A helper function to evaluate the forward transform on a given array of
        input data.

        :param X: A numpy array of shape [<any>, height, width], where height
            and width match the size of the placeholder fed to the forward
            transform.
        :param sess: Tensorflow session to use. If none is provided a temporary
            session will be used.
            
        :returns: A :py:class:`dtcwt.Pyramid` of the data. The variables in
        this pyramid will typically be only 2-dimensional (when calling the
        numpy forward transform), but these will be 3 dimensional.
        """
        if len(X.shape) == 2 and len(self.lowpass_op.get_shape()) == 3:
            logging.warn('Fed with a 2d shape input. For efficient calculation'
                        + ' feed batches of inputs.')

        lo = self._get_lowpass(X, sess)
        hi = self._get_highpasses(X, sess)
        scales = self._get_scales(X, sess)
        return Pyramid_np(lo, hi, scales)

    def eval_inv(self, Yl, Yh, sess=None):
        """
        A helper function to evaluate the inverse transform on given wavelet
        coefficients. 

        :param Yl: A numpy array of shape [<batch>, h/(2**scale), w/(2**scale)],
            where (h,w) was the size of the input image.
        :param Yh: A tuple or list of the highpass coefficients. Each entry in
            the tuple or list represents the scale the coefficients belong to. The
            size of the coefficients must match the outputs of the forward
            transform. I.e. Yh[0] should have shape [<batch>, 6, h/2, w/2], where the
            input image had shape (h, w). <batch> should be the same across all
            scales, and should match the size of the Yl first dimension.
        :param sess: Tensorflow session to use. If none is provided a temporary
            session will be used.

        :returns: A numpy array of the inverted data. 
        """
        return self._get_X(Yl, Yh, sess)

