# MATLAB support scripts

This directory contains the MATLAB files required to recreate the verification
data. Obviously you must have MATLAB installed but you also require the DT-CWT
toolbox available on [Nick Kingsbury's](http://www-sigproc.eng.cam.ac.uk/~ngk/)
home page.

The ``regen_verification.sh`` script will run MATLAB and Python to re-generate
the verification data. It uses the scripts ``gen_verif.m`` and
``verif_m_to_npz.py``. You should use this script but you may need to configure
it slightly to set the location of MATLAB on your system and the DTCWT
toolboxes.

The ``gen_verif.m`` script is not sophisticated; it simply exercises a number
of the DT-CWT toolbox routines and saves the result to ``verification.mat``.

The ``verif_m_to_npz.py`` script uses SciPy to load the MATLAB output and
convert it into NumPy's native ``.npz`` format. This file is used by the test
suite and is located at ``tests/verification.npz``.
