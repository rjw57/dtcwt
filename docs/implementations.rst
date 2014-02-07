Backend API Reference
=====================

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

