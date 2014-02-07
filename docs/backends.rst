Multiple Backend Support
========================

The ``dtcwt`` library currently provides two backends for computing the wavelet
transform: a `NumPy <http://www.numpy.org/>`_ based implementation and an OpenCL
implementation which uses the `PyOpenCL <http://mathema.tician.de/software/pyopencl/>`_
bindings for Python.

NumPy
'''''

The NumPy backend is the reference implementation of the transform. All
algorithms and transforms will have a NumPy backend. NumPy implementations are
written to be efficient but also clear in their operation.

OpenCL
''''''

Some transforms and algorithms implement an OpenCL backend. This backend, if
present, will provide an identical API to the NumPy backend. NumPy-based input
may be passed in and out of the backends but if OpenCL-based input is passed
in, a copy back to the host may be avoided in some cases. Not all transforms or
algorithms have an OpenCL-based implementation and the implementation itself
may not be full-featured.

OpenCL support depends on the `PyOpenCL
<http://mathema.tician.de/software/pyopencl/>`_ package being installed and an
OpenCL implementation being installed on your machine. Attempting to use an
OpenCL backen without both of these being present will result in a runtime (but
not import-time) exception.

Which backend should I use?
'''''''''''''''''''''''''''

The top-level transform routines, such as :py:func:`dtcwt.dtwavexfm2`, will
automatically use the NumPy backend. If you are not primarily focussed on
speed, this is the correct choice since the NumPy backend has the fullest
feature support, is the best tested and behaves correctly given single- and
double-precision input.

If you care about speed and need only single-precision calculations, the OpenCL
backend can provide significant speed-up. On the author's system, the 2D
transform sees around a times 10 speed improvement.

Using a backend
'''''''''''''''

The NumPy and OpenCL backends live in the :py:mod:`dtcwt.numpy`
and :py:mod:`dtcwt.opencl` modules respectively. Both provide
implementations of some subset of the DTCWT library functionality.

Access to the 2D transform is via a :py:class:`dtcwt.Transform2d` instance. For
example, to compute the 2D DT-CWT of the 2D real array in *X*::

    >>> from dtcwt.numpy import Transform2d
    >>> trans = Transform2d()           # You may optionally specify which wavelets to use here
    >>> Y = trans.forward(X, nlevels=4) # Perform a 4-level transform of X
    >>> imshow(Y.lowpass)               # Show coarsest scale low-pass image
    >>> imshow(Y.subbands[-1][:,:,0])   # Show first coarsest scale subband

In this case *Y* is an instance of a class which behaves like
:py:class:`dtcwt.Pyramid`. Backends are free to
return whatever result they like as long as the result can be used like this
base class. (For example, the OpenCL backend returns a
:py:class:`dtcwt.opencl.Pyramid` instance which
keeps the device-side results available.)
