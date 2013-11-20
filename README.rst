Dual-Tree Complex Wavelet Transform library for Python
======================================================

This library provides support for computing 1D, 2D and 3D dual-tree complex wavelet
transforms and their inverse in Python.
`Full documentation <https://dtcwt.readthedocs.org/>`_ is available online.

.. image:: https://travis-ci.org/rjw57/dtcwt.png?branch=master
    :target: https://travis-ci.org/rjw57/dtcwt

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

0.7.1
'''''

* Fix a memory leak in OpenCL implementation where transform results were never
  de-allocated.

.. vim:sw=4:sts=4:et
