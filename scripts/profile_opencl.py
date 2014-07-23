# A script intended to be run with cProfile to test the OpenCL transform

# Requires docopt
"""
Usage:
    profile_opencl.py [-n LEVELS] [-l ITERATIONS] [--no-copy]

Options:

    -n LEVELS           How many levels [default: 4]
    -l ITERATIONS       How many transforms [default: 1000]
    --no-copy           Do not include copy to/from device
"""

import os

import docopt
import dtcwt
import numpy as np

import pyopencl as cl
import pyopencl.array as cla

def main():
    opts = docopt.docopt(__doc__)
    nlevels = int(opts['-n'])
    transforms = int(opts['-l'])
    print('Using a {0} level transform'.format(nlevels))
    print('Looping {0} times'.format(transforms))

    lena = np.load(os.path.join(os.path.dirname(__file__), '..', 'tests', 'lena.npz'))['lena']
    lena = lena.astype(np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Plan transform
    t = dtcwt.opencl.Transform2d(queue=queue)

    if opts['--no-copy']:
        # Loop transform *not* including copy to/from device
        lena_cla = cla.to_device(queue, lena)
        for i in range(transforms):
            p = t.forward(lena_cla, nlevels=nlevels)
    else:
        # Loop transform including copy to/from device
        for i in range(transforms):
            p = t.forward(lena, nlevels=nlevels)
            _, _ = p.lowpass, p.highpasses

if __name__ == '__main__':
    main()
