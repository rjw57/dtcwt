API Reference
=============

Computing the DT-CWT
````````````````````

These functions provide API-level compatibility with MATLAB.

.. note::

    The functionality of ``dtwavexfm2b`` and ``dtwaveifm2b`` have been folded
    into ``dtwavexfm2`` and ``dtwaveifm2``. For convenience of porting MATLAB
    scripts, the original function names are available in the :py:mod:`dtcwt`
    module as aliases but they should not be used in new code.

.. automodule:: dtcwt
    :members:

NumPy
'''''

.. automodule:: dtcwt.numpy
    :members:
    :inherited-members:

OpenCL
''''''

.. automodule:: dtcwt.opencl
    :inherited-members:

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

A normal user should not need to call these functions but they are documented
here just in case you do.

.. automodule:: dtcwt.utils
    :members:

.. automodule:: dtcwt.lowlevel
    :members:
