Dual-Tree Complex Wavelet Transform library for Python
======================================================

This library provides support for computing 1D, 2D and 3D dual-tree complex wavelet
transforms and their inverse in Python.
`Full documentation <https://dtcwt.readthedocs.org/>`_ is available online.

.. image:: https://travis-ci.org/rjw57/dtcwt.png?branch=master
    :target: https://travis-ci.org/rjw57/dtcwt

.. image:: https://coveralls.io/repos/rjw57/dtcwt/badge.png?branch=master
    :target: https://coveralls.io/r/rjw57/dtcwt?branch=master
    :alt: Coverage

.. image:: https://pypip.in/license/dtcwt/badge.png
    :target: https://pypi.python.org/pypi/dtcwt/
    :alt: License

.. image:: https://pypip.in/v/dtcwt/badge.png
    :target: https://pypi.python.org/pypi/dtcwt/
    :alt: Latest Version

.. image:: https://pypip.in/d/dtcwt/badge.png
    :target: https://pypi.python.org/pypi//dtcwt/
    :alt: Downloads

Installation
````````````

The easiest way to install ``dtcwt`` is via ``easy_install`` or ``pip``::

    $ pip install dtcwt

If you want to check out the latest in-development version, look at
`the project's GitHub page <https://github.com/rjw57/dtcwt>`_. Once checked out,
installation is based on setuptools and follows the usual conventions for a
Python project::

    $ python setup.py install

(Although the `develop` command may be more useful if you intend to perform any
significant modification to the library.) A test suite is provided so that you
may verify the code works on your system::

    $ python setup.py nosetests

This will also write test-coverage information to the ``cover/`` directory.

Further documentation
`````````````````````

There is `more documentation <https://dtcwt.readthedocs.org/>`_
available online and you can build your own copy via the Sphinx documentation
system::

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/``.

Provenance
``````````

Based on the Dual-Tree Complex Wavelet Transform Pack for MATLAB by Nick
Kingsbury, Cambridge University. The original README can be found in
ORIGINAL_README.txt.  This file outlines the conditions of use of the original
MATLAB toolbox.

Changes
```````

<<<<<<< HEAD
0.9.0
'''''

0.8.0
'''''

* Verified the highpass re-sampling routines in ``dtcwt.sampling`` against the
  existing MATLAB implementation.
* Added experimental image registration routines.
* Re-organised documentation.

=======
>>>>>>> b71d984217a4cd51cd7507378a842454051acd4d
0.7.2
'''''

* Fixed regression from 0.7 where ``backend_opencl.dtwavexfm2`` would return
  ``None, None, None``.

0.7.1
'''''

* Fix a memory leak in OpenCL implementation where transform results were never
  de-allocated.

.. vim:sw=4:sts=4:et
