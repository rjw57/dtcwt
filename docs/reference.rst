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

