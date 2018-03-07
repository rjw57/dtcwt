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

.. image:: https://img.shields.io/pypi/l/dtcwt.svg
    :target: https://pypi.python.org/pypi/dtcwt/
    :alt: License

.. image:: https://img.shields.io/pypi/v/dtcwt.svg
    :target: https://pypi.python.org/pypi/dtcwt/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/dm/dtcwt.svg
    :target: https://pypi.python.org/pypi//dtcwt/
    :alt: Downloads

.. Note: this DOI link must be updated for each release.

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.9862.png
    :target: http://dx.doi.org/10.5281/zenodo.9862
    :alt: DOI: 10.5281/zenodo.9862

.. image:: https://readthedocs.org/projects/dtcwt/badge/?version=latest
    :target: https://readthedocs.org/projects/dtcwt/?badge=latest
    :alt: Documentation Status

Installation
````````````

Ubuntu 15.10 (wily) and later
'''''''''''''''''''''''''''''

Installation can be perfomed via ``apt-get``::

    $ sudo apt-get install python-dtcwt python-dtcwt-doc

The package is also currently in Debian sid (unstable).

Other operating systems
'''''''''''''''''''''''

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

    $ pip install -r tests/requirements.txt
    $ py.test

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

0.12.0
''''''

0.11.0
''''''

* Use fixed random number generator seed when generating documentation.
* Replace use of Lena image with mandrill.
* Refactor test suite to use tox + py.test.
* Documentation formatting fixes.
* Fix unsafe use of inplace casting (3D transform).
* Use explicit integer division to close #123.

0.10.1
''''''

* Fix regression in dtcwt-based image registration.
* Allow levels used for dtcwt-based image registration to be customised.

0.10.0
''''''

* Add queue parameter to low-level OpenCL ``colifilt`` and ``coldfilt`` functions.
* Significantly increase speed of ``dtcwt.registration.estimatereg`` function.
* Fix bug whereby ``dtcwt.backend_name`` was not restored when using
  ``preserve_backend_stack``.

0.9.1
'''''

* The OpenCL 2D transform was not always using the correct queue when one was
  passed explicitly.

0.9.0
'''''

* MATLAB-style functions such as ``dtwavexfm2`` have been moved into a separate
  ``dtcwt.compat`` module.
* Backends moved to ``dtcwt.numpy`` and ``dtcwt.opencl`` modules.
* Removed ``dtcwt.base.ReconstructedSignal`` which was a needless wrapper
  around NumPy arrays.
* Rename ``TransformDomainSignal`` to ``Pyramid``.
* Allow runtime configuration of default backend via ``dtcwt.push_backend`` function.
* Verified, thanks to @timseries, the NumPy 3D transform implementation against
  the MATLAB reference implementation.

0.8.0
'''''

* Verified the highpass re-sampling routines in ``dtcwt.sampling`` against the
  existing MATLAB implementation.
* Added experimental image registration routines.
* Re-organised documentation.

0.7.2
'''''

* Fixed regression from 0.7 where ``backend_opencl.dtwavexfm2`` would return
  ``None, None, None``.

0.7.1
'''''

* Fix a memory leak in OpenCL implementation where transform results were never
  de-allocated.

.. vim:sw=4:sts=4:et
