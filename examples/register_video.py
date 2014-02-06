#!/usr/bin/env python

"""
Register neighbouring frames of video and save inter-frame transform
parameters to a file.

Usage:
    register_video.py [options] <inputvideo> <outputh5>
    register_video.py (-h | --help)

Options:
    -g GROUPSIZE, --group=GROUPSIZE  Frame pairs per work-group [default: 32]

"""

import logging

import cv2
from docopt import docopt
import dtcwt
from dtcwt.numpy import Transform2d
import dtcwt.registration as reg
import dtcwt.sampling
import numpy as np
from mpi4py import MPI
import tables

# Parse command line options
OPTS = docopt(__doc__)

class VideoReader(object):
    def __init__(self, filename, groupsize=None):
        self._vc = cv2.VideoCapture(filename)
        self._last_frame = None
        self._last_frame_idx = -1

        self.groupsize = groupsize or int(OPTS['--group'])

    def read_next_gof(self):
        """
        Read the next group of frames from *videoreader*.

        """

        frames = []
        if self._last_frame is not None:
            frames.append((self._last_frame_idx, self._last_frame))

        for it in xrange(self.groupsize):
            success, f = self._vc.read()
            if not success:
                break

            self._last_frame = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) / 255.0
            self._last_frame_idx += 1

            frames.append((self._last_frame_idx, self._last_frame))

        return frames

def avecs_for_frames(frames):
    if len(frames) <= 1:
        return [],[]

    avecs = []
    trans = Transform2d()

    idx, frame = frames[0]
    t = trans.forward(frame, nlevels=5)

    h_pair = (None, t)
    idx_pair = (None, idx)
    idx_pairs = []

    for idx, frame in frames[1:]:
        t = trans.forward(frame, nlevels=5)

        h_pair = (h_pair[1], t)
        idx_pair = (idx_pair[1], idx)

        idx_pairs.append(idx_pair)
        avecs.append(reg.estimatereg(h_pair[0], h_pair[1]))
        logging.info('Finished frame pair {0}'.format(idx_pair))

    return idx_pairs, avecs

class Metadata(tables.IsDescription):
    videopath = tables.StringCol(512)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(level=logging.INFO, format='Rank ' + str(comm.Get_rank()) + ': %(message)s')

    logging.info('Launched')

    if rank == 0:
        # I'm rank 0. Open the video file for reading
        logging.info('Loading video from "{0}"'.format(OPTS['<inputvideo>']))
        v = VideoReader(OPTS['<inputvideo>'])
        outfile = tables.openFile(OPTS['<outputh5>'], mode='w',
            title='Frame to frame registration from "{0}"'.format(OPTS['<inputvideo>']))

        # Create a frame pair e-array
        frame_pairs = outfile.createEArray('/', 'frame_idx_pairs',
                                atom=tables.Int64Atom(),
                                shape=(0,2),
                                title='(before, after) frame indices for corresponding affine parameters')

        metadata_table = outfile.createTable('/', 'metadata', Metadata, 'Metadata')
        metadata = metadata_table.row
        metadata['videopath'] = OPTS['<inputvideo>']
        metadata.append()

        affine_params = None # Will be created after first bit of data

        # Commit to disk
        outfile.flush()

    is_last_iteration = False
    last_frame = None
    group_idx = 0
    while not is_last_iteration:
        groups = None

        if rank == 0:
            logging.info('Reading next set of work group frames')

            # Read frames for next work group
            groups = []
            for group in xrange(size):
                gof = v.read_next_gof()
                groups.append((group_idx, gof))
                group_idx += 1

                # If we run out of frames, this is the last iteration
                is_last_iteration = len(gof) <= 1

            logging.info('Sending work group frames')

        # Broadcast iteration flag to nodes
        is_last_iteration = comm.bcast(is_last_iteration, root=0)

        # Scatter to nodes
        group_id, frames = comm.scatter(groups, root=0)

        logging.info('received work group id={0} of {1} frame(s)'.format(group_id, len(frames)))

        # Calculate result
        idxs_and_avecs = avecs_for_frames(frames)

        # Send result back to rank 0
        logging.info('finished. Sending {1} results'.format(rank, len(idxs_and_avecs)))
        gathered = comm.gather(idxs_and_avecs, root=0)

        if rank == 0:
            for idxs, av in gathered:
                assert len(idxs) == len(av)

                if affine_params is None and len(av) > 0:
                    affine_params = outfile.createEArray('/', 'affine_parameters',
                                    atom=tables.Float64Atom(),
                                    shape=(0,) + av[0].shape,
                                    title='affine parameters for corresponding frame index pairs')

                if len(av) > 0:
                    affine_params.append(av)
                    frame_pairs.append(idxs)

            outfile.flush()

            logging.info('Frame pairs processed: {0}'.format(affine_params.shape[0]))

    if rank == 0:
        outfile.close()

if __name__ == '__main__':
    main()
