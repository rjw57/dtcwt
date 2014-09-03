import numpy as np

from .util import assert_almost_equal, skip_if_no_cl
import tests.datasets as datasets

# OpenCL is optional since skip_if_no_cl is used.
try:
    import pyopencl as cl
    import pyopencl.array as cla
except ImportError:
    pass

@skip_if_no_cl
def setup():
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
def test_edge_reflect():
    """Tests that the edge (rather than centre) of pixel reflection sampling works."""
    from dtcwt.opencl.convolve import _write_input_pixel_test_image
    output_array = cla.zeros(queue, (128,128), cla.vec.float2)
    _write_input_pixel_test_image(queue, output_array, (-30,-40), (11,12)).wait()
    output_array = output_array.get()

    assert output_array['x'].min() == 0
    assert output_array['y'].min() == 0
    assert output_array['x'].max() == 10
    assert output_array['y'].max() == 11

    assert output_array[30,40]['x'] == 0
    assert output_array[30,40]['y'] == 0
    assert output_array[29,40]['x'] == 0
    assert output_array[29,40]['y'] == 0
    assert output_array[30,39]['x'] == 0
    assert output_array[30,39]['y'] == 0
    assert output_array[29,39]['x'] == 0
    assert output_array[29,39]['y'] == 0

    assert output_array[31,40]['x'] == 1
    assert output_array[31,40]['y'] == 0
    assert output_array[30,41]['x'] == 0
    assert output_array[30,41]['y'] == 1
    assert output_array[31,41]['x'] == 1
    assert output_array[31,41]['y'] == 1

    assert output_array[40,51]['x'] == 10
    assert output_array[40,51]['y'] == 11
    assert output_array[41,51]['x'] == 10
    assert output_array[41,51]['y'] == 11
    assert output_array[40,52]['x'] == 10
    assert output_array[40,52]['y'] == 11
    assert output_array[41,52]['x'] == 10
    assert output_array[41,52]['y'] == 11

@skip_if_no_cl
def test_trivial_convolution():
    """Tests convolving an input with a single coefficient kernel."""
    from dtcwt.opencl.convolve import Convolution1D
    coeffs = np.empty((1,), cla.vec.float2)
    coeffs['x'] = 1
    coeffs['y'] = 0.5
    convolution = Convolution1D(queue, coeffs)

    # Test each plane
    output_device = cla.empty(queue, traffic_rgb.shape[:2], cla.vec.float2)
    for plane_idx in range(traffic_rgb.shape[2]):
        print('Testing plane {0}/{1}'.format(plane_idx+1, traffic_rgb.shape[2]))
        plane = traffic_rgb[:,:,plane_idx]
        convolution(plane, output_device).wait()

        output = output_device.get()
        plane_host = traffic_rgb.get()[:,:,plane_idx]

        assert_almost_equal(output['x'], plane_host)
        assert_almost_equal(output['y'], 0.5 * plane_host)

@skip_if_no_cl
def test_low_highpass():
    """Tests convolving an input with a less trivial kernel."""
    from dtcwt.opencl.convolve import Convolution1D, biort
    from dtcwt.numpy.lowlevel import colfilter

    # Get a low- and highpass kernel
    kernel_coeffs = biort('near_sym_b')
    assert kernel_coeffs.shape[0] > 1

    # Prepare the convolution
    convolution = Convolution1D(queue, kernel_coeffs)

    # Test each plane
    output_device = cla.empty(queue, traffic_rgb.shape[:2], cla.vec.float2)
    for plane_idx in range(traffic_rgb.shape[2]):
        print('Testing plane {0}/{1}'.format(plane_idx+1, traffic_rgb.shape[2]))
        plane = traffic_rgb[:,:,plane_idx]
        convolution(plane, output_device).wait()

        output = output_device.get()

        plane_host = traffic_rgb.get()[:,:,plane_idx]
        gold_low = colfilter(plane_host, kernel_coeffs['x'])
        gold_high = colfilter(plane_host, kernel_coeffs['y'])

        assert_almost_equal(gold_low, output['x'])
        assert_almost_equal(gold_high, output['y'])
