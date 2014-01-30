Getting Started
===============

This section will guide you through using the ``dtcwt`` library.  Once
installed, you are most likely to use one of these functions:

* :py:func:`dtcwt.dtwavexfm` -- 1D DT-CWT transform.
* :py:func:`dtcwt.dtwaveifm` -- Inverse 1D DT-CWT transform.
* :py:func:`dtcwt.dtwavexfm2` -- 2D DT-CWT transform.
* :py:func:`dtcwt.dtwaveifm2` -- Inverse 2D DT-CWT transform.
* :py:func:`dtcwt.dtwavexfm3` -- 3D DT-CWT transform.
* :py:func:`dtcwt.dtwaveifm3` -- Inverse 3D DT-CWT transform.

See :doc:`reference` for full details on how to call these functions. We shall
present some simple usage below.

Installation
------------

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

Building the documentation
``````````````````````````

There is `a pre-built <https://dtcwt.readthedocs.org/>`_ version of this
documentation available online and you can build your own copy via the Sphinx
documentation system:

.. code-block:: console

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/``.
