# MATLAB support scripts

This directory contains the MATLAB files required to recreate the verification
data. Obviously you must have MATLAB installed but you also require the DT-CWT
toolbox available on [Nick Kingsbury's](http://www-sigproc.eng.cam.ac.uk/~ngk/)
home page.

The ``gen_verif.m`` script is not sophisticated; they simply exercise a number
of the DT-CWT toolbox routines and saves the result to ``verification.mat``.
Run it with a command like the following:

```console
$ MATLABPATH=/path/to/dtcwt_toolbox4_3 /path/to/matlab -nosplash -nodesktop -r "run /path/to/gen_verif; quit"
```

The ``verif_m_to_npz.py`` script uses SciPy to load the MATLAB output and
convert it into NumPy's native ``.npz`` format. This file is used by the test
suite and is located at ``tests/verification.npz``.
