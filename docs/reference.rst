API Reference
=============

Main interface
``````````````

.. automodule:: dtcwt
    :members:

.. automodule:: dtcwt.coeffs
    :members:

Keypoint analysis
`````````````````

.. automodule:: dtcwt.keypoint
    :members:

Image sampling
``````````````

.. automodule:: dtcwt.sampling
    :members:

Image registration
``````````````````

.. automodule:: dtcwt.registration
    :members:

Plotting functions
``````````````````

.. automodule:: dtcwt.plotting
    :members:

Miscellaneous and low-level support functions
`````````````````````````````````````````````

.. automodule:: dtcwt.utils
    :members:

Compatibility with MATLAB
`````````````````````````

.. automodule:: dtcwt.compat
    :members:

Backends
````````

The following modules provide backend-specific implementations. Usually you
won't need to import these modules directly; the main API will use an
appropriate implementation. Occasionally, however, you may want to benchmark
one implementation against the other.

NumPy
'''''

.. automodule:: dtcwt.numpy
    :members:
    :inherited-members:

.. automodule:: dtcwt.numpy.lowlevel
    :members:

OpenCL
''''''

.. automodule:: dtcwt.opencl
    :members:
    :inherited-members:

.. automodule:: dtcwt.opencl.lowlevel
    :members:

.. _tensorflowbackend:

Tensorflow
''''''''''
Currently the Tensorflow backend only supports single precision operations, and
only has functionality for the Transform1d() and Transform2d() classes (i.e. 
changing the backend to 'tf' will still use the numpy Transform3d() class).

To preserve functionality, the Transform1d() and Transform2d() classes have
a `forward` method which behaves identically to the NumPy backend. However, to
get speedups with tensorflow, we want to feed our transform batches of images.
For this reason, the 1-D and 2-D transforms also have `forward_channels` and
`inverse_channels` methods. See the below documentation for how to use these.

.. automodule:: dtcwt.tf
    :members:
    :inherited-members:

.. automodule:: dtcwt.tf.lowlevel
    :members:
