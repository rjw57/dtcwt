The dtcwt library
=================

This library provides support for computing 1D, 2D and 3D dual-tree complex
wavelet transforms and their inverse in Python. The interface is simple and
easy to use. As a quick example, a 1D DT-CWT can be performed from the Python
console in a single line::

    >>> import dtcwt
    >>> Yl, Yh = dtcwt.dtwavexfm([1,2,3,4], nlevels=3) # 3 levels, default wavelets

The interface is intentionally similar to the existing MATLAB dual-tree complex
wavelet transform toolbox provided by `Prof. Nick Kingsbury
<http://www-sigproc.eng.cam.ac.uk/~ngk/>`_. This library is intended to ease
the porting of algorithms written using the original MATLAB toolbox to Python.

Features of note
````````````````

The features of the ``dtcwt`` library are:

* 1D, 2D and 3D forward and inverse Dual-tree Complex Wavelet Transform
  implementations.
* API similarity with the DTCW MATLAB toolbox.
* Automatic selection of single versus double precision calculation.
* Built-in support for the most common complex wavelet families.

Installation
````````````

The easiest way to install ``dtcwt`` is via ``easy_install`` or ``pip``:

.. code-block:: console

    $ pip install dtcwt

If you want to check out the latest in-development version, look at
`the project's GitHub page <https://github.com/rjw57/dtcwt>`_. Once checked out,
installation is based on setuptools and follows the usual conventions for a
Python project:

.. code-block:: console

    $ python setup.py install

(Although the `develop` command may be more useful if you intend to perform any
significant modification to the library.) A test suite is provided so that you
may verify the code works on your system:

.. code-block:: console

    $ python setup.py nosetests

This will also write test-coverage information to the ``cover/`` directory.

Further documentation
`````````````````````

There is `more documentation <https://dtcwt.readthedocs.org/>`_
available online and you can build your own copy via the Sphinx documentation
system::

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/``.

Licence
```````

The original toolbox is copyrighted and there are some restrictions on use
which are outlined in the file
:download:`ORIGINAL_README.txt<../ORIGINAL_README.txt>`.
Aside from portions directly derived from the original MATLAB toolbox, any
additions in this library and this documentation are licensed under the
2-clause BSD licence as documented in the file
:download:`COPYING.txt<../COPYING.txt>`.

Contents
`````````````````

.. toctree::
    :maxdepth: 1

    gettingstarted
    examples
    reference

.. vim:sw=4:sts=4:et
