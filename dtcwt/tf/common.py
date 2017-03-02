from __future__ import absolute_import

from dtcwt.numpy import Pyramid
import tensorflow as tf

class Pyramid_tf(object):
    """A representation of a transform domain signal.
    Backends are free to implement any class which respects this interface for
    storing transform-domain signals. The inverse transform may accept a
    backend-specific version of this class but should always accept any class
    which corresponds to this interface.
    .. py:attribute:: lowpass
        A NumPy-compatible array containing the coarsest scale lowpass signal.
    .. py:attribute:: highpasses
        A tuple where each element is the complex subband coefficients for
        corresponding scales finest to coarsest.
    .. py:attribute:: scales
        *(optional)* A tuple where each element is a NumPy-compatible array
        containing the lowpass signal for corresponding scales finest to
        coarsest. This is not required for the inverse and may be *None*.
    """
    def __init__(self, lowpass, highpasses, scales=None):
        self.lowpass = lowpass
        self.highpasses = highpasses
        self.scales = scales
        
    def eval(self, sess, placeholder, data):
        try:
            lo = sess.run(self.lowpass, {placeholder : data})
            hi = sess.run(self.highpasses, {placeholder : data})
            if self.scales is not None:
                scales = sess.run(self.scales, {placeholder : data})
            else:
                scales = None
        except ValueError:
            lo = sess.run(self.lowpass, {placeholder : [data]})
            hi = sess.run(self.highpasses, {placeholder : [data]})
            if self.scales is not None:
                scales = sess.run(self.scales, {placeholder : [data]})
            else:
                scales = None


        return Pyramid(lo, hi, scales)
    
