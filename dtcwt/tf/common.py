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

    def _get_lowpass(self, data):
        if self.lowpass_op is None:
            return None
        with tf.Session(graph=self.graph) as sess:
            try:
                y = sess.run(self.lowpass_op, {self.X : data})
            except ValueError:
                y = sess.run(self.lowpass_op, {self.X : [data]})[0]
        return y
        
    def _get_highpasses(self, data):
        if self.highpasses_ops is None:
            return None
        with tf.Session(graph=self.graph) as sess:
            try: 
                y = tuple(
                        [sess.run(layer_hp, {self.X : data}) 
                        for layer_hp in self.highpasses_ops])
            except ValueError:
                y = tuple(
                        [sess.run(layer_hp, {self.X : [data]})[0] 
                        for layer_hp in self.highpasses_ops])
        return y

    def _get_scales(self, data):
        if self.scales_ops is None:
            return None
        with tf.Session(graph=self.graph) as sess:
            try:
                y = tuple(
                        sess.run(layer_scale, {self.X : data})
                        for layer_scale in self.scales_ops)
            except ValueError:
                y = tuple(
                        sess.run(layer_scale, {self.X : [data]})[0]
                        for layer_scale in self.scales_ops)
        return y

    def _get_X(self, Yl, Yh):
        if self.X is None:
            return None
        with tf.Session(graph=self.graph) as sess:
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

    def eval_fwd(self, X):
        lo = self._get_lowpass(X)
        hi = self._get_highpasses(X)
        scales = self._get_scales(X)
        return Pyramid_np(lo, hi, scales)

    def eval_inv(self, Yl, Yh):
        return self._get_X(Yl, Yh)

