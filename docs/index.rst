The dtcwt library
=================

The ``dtcwt`` library provides a Python implementation of the 1, 2 and 3-D
dual-tree complex wavelet transform along with some associated algorithms. It
contains a pure CPU implementation which makes use of NumPy along with an
accelerated GPU implementation using OpenCL.

Comparison with MATLAB toolbox
------------------------------

The canonical implementation of the DT-CWT is that provided by Professor Nick
Kingsbury on `his website <http://www-sigproc.eng.cam.ac.uk/Main/NGK>`_. This
library aims to have near-identical output (to within a small multiple of
machine precision). Significant deviation is a bug and should be `reported
<https://github.com/rjw57/dtcwt/issues>`_. Cross-verification of the transform
output is part of the test suite and each and every change is checked against
that test suite automatically.

It is hoped that testing this will allow confidence in this library being
suitable for porting existing MATLAB scripts over to Python. To that end there
is a :py:mod:`dtcwt.compat` module which provides an API similar to the original
MATLAB toolbox.

Contents
--------

.. toctree::
    :maxdepth: 2

    gettingstarted
    transforms
    backends
    algorithms
    examples
    reference

.. vim:sw=4:sts=4:et
