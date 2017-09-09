from __future__ import absolute_import

try:
    import tensorflow as tf
except ImportError:
    # The lack of tensorflow will be caught by the low-level routines.
    pass


class Pyramid(object):
    """A tensorflow representation of a transform domain signal.

    Backends are free to implement any class which respects this interface for
    storing transform-domain signals, so long as the attributes have the
    correct names and are tensorflow tensors (or placeholders).
    The inverse transform may accept a backend-specific version of this class
    but should always accept any class which corresponds to this interface.

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
    """
    def __init__(self, lowpass, highpasses, scales=None, numpy=False):
        self.lowpass_op = lowpass
        self.highpasses_ops = highpasses
        self.scales_ops = scales
        self.numpy = numpy

    @property
    def lowpass(self):
        if not hasattr(self, '_lowpass'):
            if self.lowpass_op is None:
                self._lowpass = None
            else:
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    self._lowpass = sess.run(self.lowpass_op)
        return self._lowpass

    @property
    def highpasses(self):
        if not hasattr(self, '_highpasses'):
            if self.highpasses_ops is None:
                self._highpasses = None
            else:
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    self._highpasses = \
                        tuple(sess.run(x) for x in self.highpasses_ops)
        return self._highpasses

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            if self.scales_ops is None:
                self._scales = None
            else:
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    self._scales = tuple(sess.run(x) for x in self.scales_ops)
        return self._scales
