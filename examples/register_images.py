#!/usr/bin/env python

"""
Register neighbouring frames of video and save inter-frame transform
parameters to a file.

Usage:
    register_images.py [options] <prevframe> <nextframe> <output>
    register_images.py (-h | --help)

Options:

    --cl        Attempt to use OpenCL where possible.

"""

import logging

from docopt import docopt
import dtcwt
from dtcwt.opencl import Transform2d as CLTransform2d
from dtcwt.numpy import Transform2d as NumPyTransform2d
import dtcwt.registration as reg
import dtcwt.sampling
from PIL import Image # Use 'Pillow', the PIL fork
import numpy as np
import tables

# Parse command line options
OPTS = docopt(__doc__)

# Set logging options
logging.basicConfig(level=logging.INFO)

def avecs_for_pair(prev, next_):
    trans = CLTransform2d() if OPTS['--cl'] else NumPyTransform2d()
    t1 = trans.forward(prev, nlevels=5)
    t2 = trans.forward(next_, nlevels=5)
    return reg.estimatereg(t1, t2)

class Metadata(tables.IsDescription):
    previmpath = tables.StringCol(512)
    nextimpath = tables.StringCol(512)

def main():
    logging.info('Launched')

    logging.info('Loading "prev" image from "{0}"'.format(OPTS['<prevframe>']))
    pim = np.array(Image.open(OPTS['<prevframe>']).convert('L')) / 255.0

    logging.info('Loading "next" image from "{0}"'.format(OPTS['<nextframe>']))
    nim = np.array(Image.open(OPTS['<nextframe>']).convert('L')) / 255.0

    logging.info('Estimating registration')
    avecs = avecs_for_pair(pim, nim)

    logging.info('Calculating velocity field')
    vxs, vys = dtcwt.registration.velocityfield(avecs, avecs.shape[:2], method='bilinear')

    logging.info('Saving result to {0}'.format(OPTS['<output>']))
    np.savez_compressed(OPTS['<output>'], avecs=avecs, vxs=vxs, vys=vys)

if __name__ == '__main__':
    main()
