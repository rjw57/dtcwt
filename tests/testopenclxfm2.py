import numpy as np

from .util import assert_almost_equal, skip_if_no_cl, assert_pyramids_almost_equal
import tests.datasets as datasets

# OpenCL is optional since skip_if_no_cl is used.
try:
    import pyopencl as cl
    import pyopencl.array as cla
    HAVE_OPENCL=True
except ImportError:
    HAVE_OPENCL=False

def setup():
    if not HAVE_OPENCL:
        return

    global traffic_rgb, queue, ctx
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    print('Using context: {0}'.format(ctx))
    traffic_rgb = cla.to_device(queue, datasets.traffic_hd_rgb().astype(np.float32))

@skip_if_no_cl
def test_context_and_queue():
    assert ctx is not None
    assert queue is not None

@skip_if_no_cl
def test_level1_forward_transform():
    from dtcwt.opencl.transform2d import Transform2d
    from dtcwt.numpy.transform2d import Transform2d as GoldTransform2d

    t = Transform2d(queue=queue)
    gold_t = GoldTransform2d()

    for plane_idx in range(traffic_rgb.shape[2]):
        print('Testing plane {0}/{1}'.format(plane_idx+1, traffic_rgb.shape[2]))

        plane_device = traffic_rgb[:,:,plane_idx]
        plane_host = traffic_rgb.get()[:,:,plane_idx]

        # Perform gold level 1 transform
        gold = gold_t.forward(plane_host, nlevels=1)

        # Perform opencl level 1 transform
        ocl = t.forward(plane_device, nlevels=1)

        # Compare result
        assert_pyramids_almost_equal(gold, ocl)
