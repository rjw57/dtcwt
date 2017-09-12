Multiple Backend Support
========================


The ``dtcwt`` library currently provides three backends for computing the wavelet
transform: a `NumPy <http://www.numpy.org/>`_ based implementation, an OpenCL
implementation which uses the `PyOpenCL <http://mathema.tician.de/software/pyopencl/>`_
bindings for Python, and a Tensorflow implementation which uses the 
`Tensorflow <https://www.tensorflow.org>`_ bindings for Python.

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
OpenCL backend without both of these being present will result in a runtime (but
not import-time) exception.

Tensorflow
''''''''''

If you want to take advantage of having a GPU on your machine, 
some transforms and algorithms have been implemented with a Tensorflow backend.
This backend will provide an identical API to the NumPy backend.
I.e. NumPy-based input may be passed to a tensorflow backend in the same manner
as it was passed to the NumPy backend. In which case it
will be converted to a tensorflow variable, the transform performed, and then
converted back to a NumPy variable afterwards. This conversion between types can
be avoided if a tensorflow variable is passed to the dtcwt Transforms.

The real speedup gained from using GPUs is obtained by parallel processing. For
this reason, when using the tensorflow backend, the Transforms can accept
batches of images. To do this, see the `forward_channels` and `inverse_channels`
methods. More information is in the :ref:`tensorflowbackend` section.

Tensorflow support depends on the 
`Tensorflow <https://www.tensorflow.org/install/>`_ python package being installed in the
current python environment, as well as the necessary CUDA + CUDNN libraries
installed). Attempting to use a Tensorflow backend without the python package
available will result in a runtime (but not import-time) exception. Attempting
to use the Tensorflow backend without the CUDA and CUDNN libraries properly
installed and linked will result in the Tensorflow backend being used, but
operations will be run on the CPU rather than the GPU.

If you do not have a GPU, some speedup can still be seen for using Tensorflow with
the CPU vs the plain NumPy backend, as tensorflow will naturally use multiple
processors.  

Which backend should I use?
'''''''''''''''''''''''''''

The top-level transform routines, such as :py:class:`dtcwt.Transform2d`, will
automatically use the NumPy backend. If you are not primarily focussed on
speed, this is the correct choice since the NumPy backend has the fullest
feature support, is the best tested and behaves correctly given single- and
double-precision input.

If you care about speed and need only single-precision calculations, the OpenCL
or Tensorflow backends can provide significant speed-up. 
On the author's system, the 2D transform sees around a times 10 speed
improvement for the OpenCL backend, and a 8-10 times speed up for the Tensorflow
backend.

Using a backend
'''''''''''''''

The NumPy, OpenCL and Tensorflow backends live in the :py:mod:`dtcwt.numpy`,
:py:mod:`dtcwt.opencl`, and :py:mod:`dtcwt.tf` modules respectively. All provide
implementations of some subset of the DTCWT library functionality.

Access to the 2D transform is via a :py:class:`dtcwt.Transform2d` instance. For
example, to compute the 2D DT-CWT of the 2D real array in *X*::

    >>> from dtcwt.numpy import Transform2d
    >>> trans = Transform2d()           # You may optionally specify which wavelets to use here
    >>> Y = trans.forward(X, nlevels=4) # Perform a 4-level transform of X
    >>> imshow(Y.lowpass)               # Show coarsest scale low-pass image
    >>> imshow(Y.highpasses[-1][:,:,0])   # Show first coarsest scale subband

In this case *Y* is an instance of a class which behaves like
:py:class:`dtcwt.Pyramid`. Backends are free to
return whatever result they like as long as the result can be used like this
base class. (For example, the OpenCL backend returns a
:py:class:`dtcwt.opencl.Pyramid` instance which
keeps the device-side results available.)

The default backend used by :py:class:`dtcwt.Transform2d`, etc can be
manipulated using the :py:func:`dtcwt.push_backend` function. For example, to
switch to the OpenCL backend

.. code-block:: python

    dtcwt.push_backend('opencl')
    xfm = Transform2d()
    # ... Transform2d, etc now use OpenCL ...

and to switch to the Tensorflow backend

.. code-block:: python
    
    dtcwt.push_backend('tf')
    xfm = Transform2d()
    # ... Transform2d, etc now use Tensorflow ...

As is suggested by the name, changing the backend manipulates a stack behind the
scenes and so one can temporarily switch backend using
:py:func:`dtcwt.push_backend` and :py:func:`dtcwt.pop_backend`

.. code-block:: python

    # Run benchmark with NumPy
    my_benchmarking_function()

    # Run benchmark with OpenCL
    dtcwt.push_backend('opencl')
    my_benchmarking_function()
    dtcwt.pop_backend()

It is safer to use the :py:func:`dtcwt.preserve_backend_stack` function. This
returns a guard object which can be used with the ``with`` statement to save
the state of the backend stack

.. code-block:: python

    with dtcwt.preserve_backend_stack():
        dtcwt.push_backend('opencl')
        my_benchmarking_function()

    # Outside of the 'with' clause the backend is reset to numpy.

Finally the default backend may be set via the ``DTCWT_BACKEND`` environment
variable. This is useful to run scripts with different backends without having
to modify their source.
