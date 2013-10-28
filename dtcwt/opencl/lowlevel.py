import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from memoized import memoized
from six.moves import xrange
import struct

from dtcwt.lowlevel import asfarray, as_column_vector, reflect, _column_convolve

def colfilter(X, h):
    """Filter the columns of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [1 0].

    The filtering will be accelerated via OpenCL.

    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    
    # Interpret all inputs as arrays
    X = asfarray(X)
    h = as_column_vector(h)

    return to_array(axis_convolve(X, h))

def coldfilt(X, ha, hb):
    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b   
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b       
        -q filt on o          a       a       b       b       b       b       a       a

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  Symmetric
    extension with repeated end samples is used on the composite X columns
    before each filter is applied.

    Raises ValueError if the number of rows in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    queue = get_default_queue()

    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    # Perform filtering on columns of extended matrix X(xe,:) in 4 ways.
    Y = axis_convolve_dfilter(X, ha, queue=queue)

    return to_array(Y)

def colifilt(X, ha, hb):
    """ Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e `:math:`|h(m/2)| >
    |h(m/2 + 1)|`).
    
    .. code-block:: text

                          ext       left edge                      right edge       ext
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b       
        -q filt on o          a       a       b       b       b       b       a       a
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b   
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
   
    The output is interpolated by two from the input sample rate and the
    results from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.
   
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2')

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same')

    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even')

    m = ha.shape[0]
    m2 = np.fix(m*0.5)

    Y = np.zeros((r*2,c), dtype=X.dtype)
    if not np.any(np.nonzero(X[:])[0]):
        return Y

    if m2 % 2 == 0:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)
       
        t = np.arange(3, r+m, 2)
        if np.sum(ha*hb) > 0:
            ta = t
            tb = t - 1
        else:
            ta = t - 1
            tb = t
       
        # Select odd and even samples from ha and hb. Note that due to 0-indexing
        # 'odd' and 'even' are not perhaps what you might expect them to be.
        hao = as_column_vector(ha[0:m:2])
        hae = as_column_vector(ha[1:m:2])
        hbo = as_column_vector(hb[0:m:2])
        hbe = as_column_vector(hb[1:m:2])
       
        s = np.arange(0,r*2,4)
       
        Y[s,:]   = _column_convolve(X[xe[tb-2],:],hae)
        Y[s+1,:] = _column_convolve(X[xe[ta-2],:],hbe)
        Y[s+2,:] = _column_convolve(X[xe[tb  ],:],hao)
        Y[s+3,:] = _column_convolve(X[xe[ta  ],:],hbo)
    else:
        # m/2 is odd, so set up t to start on b samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)

        t = np.arange(2, r+m-1, 2)
        if np.sum(ha*hb) > 0:
            ta = t
            tb = t - 1
        else:
            ta = t - 1
            tb = t
       
        # Select odd and even samples from ha and hb. Note that due to 0-indexing
        # 'odd' and 'even' are not perhaps what you might expect them to be.
        hao = as_column_vector(ha[0:m:2])
        hae = as_column_vector(ha[1:m:2])
        hbo = as_column_vector(hb[0:m:2])
        hbe = as_column_vector(hb[1:m:2])
       
        s = np.arange(0,r*2,4)
       
        Y[s,:]   = _column_convolve(X[xe[tb],:],hao)
        Y[s+1,:] = _column_convolve(X[xe[ta],:],hbo)
        Y[s+2,:] = _column_convolve(X[xe[tb],:],hae)
        Y[s+3,:] = _column_convolve(X[xe[ta],:],hbe)

    return Y

@memoized
def get_default_queue():
    """Return the default queue used for computation if one is not specified.

    This function is memoized and so only one queue is created after multiple invocations.
    """
    ctx = cl.create_some_context()
    return cl.CommandQueue(ctx)

def _to_queue(queue):
    if queue is not None:
        return queue
    return get_default_queue()

def _to_device(X, queue=None):
    if isinstance(X, cl_array.Array) and X.queue is queue:
        return X
    return cl_array.to_device(_to_queue(queue), np.array(X, dtype=np.float32))

def to_array(a, queue=None):
    queue = queue or a.queue or _to_queue(queue)
    rv = np.empty(a.shape, a.dtype)
    cl.enqueue_copy(queue, rv, a.data).wait()
    return rv

def _apply_kernel(X, h, kern, output, axis=0, elementstep=1):
    queue = _to_queue(output.queue)

    # If necessary, convert X and h to device arrays
    h_device = _to_device(h, queue)
    X_device = _to_device(X, queue)

    # Work out size of work group taking into account element step
    work_shape = np.array(output.shape[:3])
    work_shape[axis] /= elementstep

    # Work out optimum group size
    if work_shape.shape[0] >= 2 and np.all(work_shape[:2] > 1):
        local_shape = (int(np.floor(np.sqrt(queue.device.max_work_group_size))),) * 2 + (1,)
        local_shape = local_shape[:len(output.shape)]
    else:
        local_shape = (queue.device.max_work_group_size, 1, 1)

    global_shape = list(int(np.ceil(x/float(y))*y) for x, y in zip(work_shape, local_shape))

    X_strides = struct.pack('iiii', *(tuple(s/X_device.dtype.itemsize for s in X_device.strides) + (0,0,0,0))[:4])
    X_shape = struct.pack('iiii', *(tuple(X_device.shape) + (1,1,1,1))[:4])
    X_offset = np.int32(X_device.offset)

    Y_strides = struct.pack('iiii', *(tuple(s/output.dtype.itemsize for s in output.strides) + (0,0,0,0))[:4])
    Y_shape = struct.pack('iiii', *(tuple(output.shape) + (1,1,1,1))[:4])
    Y_offset = np.int32(output.offset)

    h_stride = np.int32(h_device.strides[0] / h_device.dtype.itemsize)
    h_shape = np.int32(h_device.shape[0])
    h_offset = np.int32(h_device.offset)

    # Perform actual convolution
    kern(queue, global_shape, local_shape,
            X_device.base_data, X_strides, X_shape, X_offset,
            h_device.base_data, h_stride, h_shape, h_offset,
            output.base_data, Y_strides, Y_shape, Y_offset,
            np.int32(axis))

    return output

def axis_convolve(X, h, axis=0, queue=None, output=None):
    """Filter along an of *X* using filter vector *h*.  If *h* has odd length, each
    output sample is aligned with each input sample and *Y* is the same size as
    *X*.  If *h* has even length, each output sample is aligned with the mid point
    of each pair of input samples, and the output matrix's shape is increased
    by one along the convolution axis.

    After convolution, the :pyclass:`pyopencl.array.Array` instance holding the
    device-side output is returned. This may be accessed on the host via
    :pyfunc:`to_array`.

    The axis of convolution is specified by *axis*. The default direction of
    convolution is column-wise.

    If *queue* is non-``None``, it should be a :pyclass:`pyopencl.CommandQueue`
    instance which is used to perform the computation. If ``None``, a default
    global queue is used.
    
    If *output* is non-``None``, it should be a :pyclass:`pyopencl.array.Array`
    instance which the result is written into. If ``None``, an output array is
    created.
    """

    queue = _to_queue(queue)
    kern = _convolve_kernel_for_queue(queue.context)

    # Create output if not specified
    if output is None:
        output_shape = list(X.shape)
        if h.shape[0] % 2 == 0:
            output_shape[axis] += 1
        output = cl_array.empty(queue, output_shape, np.float32)

    return _apply_kernel(X, h, kern, output, axis=axis)

def axis_convolve_dfilter(X, h, axis=0, queue=None, output=None):
    queue = _to_queue(queue)
    kern = _dfilter_kernel_for_queue(queue.context)

    # Create output if not specified
    if output is None:
        output_shape = list(X.shape)
        output_shape[axis] >>= 1
        output = cl_array.zeros(queue, output_shape, np.float32)

    return _apply_kernel(X, h, kern, output, axis=axis, elementstep=2)

@memoized
def _convolve_kernel_for_queue(context):
    """Return a kernel for convolution suitable for use with *context*. The
    return values are memoized.
    
    """
    kern_prog = cl.Program(context, CL_ARRAY_HEADER + CONVOLVE_KERNEL)
    kern_prog.build()
    return kern_prog.convolve_kernel

@memoized
def _dfilter_kernel_for_queue(context):
    """Return a kernel for convolution suitable for use with *context*. The
    return values are memoized.
    
    """
    kern_prog = cl.Program(context, CL_ARRAY_HEADER + DFILTER_KERNEL)
    kern_prog.build()
    return kern_prog.convolve_kernel

# Functions to access OpenCL Arrays within a kernel
CL_ARRAY_HEADER = '''
struct array_spec
{
    int4 strides;
    int4 shape;
    int offset;
};

inline int coord_to_offset(int4 coord, struct array_spec spec)
{
    int4 m = spec.strides * coord;
    return spec.offset + m.x + m.y + m.z + m.w;
}

// magic function to reflect the sampling co-ordinate about the
// *outer edges* of pixel co-ordinates x_min, x_max. The output will 
// always be in the range (x_min, x_max].
int4 reflect(int4 x, int4 x_min, int4 x_max)
{
    int4 rng = x_max - x_min;
    int4 rng_by_2 = 2 * rng;
    int4 mod = (x - x_min) % rng_by_2;
    int4 normed_mod = select(mod, mod + rng_by_2, mod < 0);

    int4 y1 = convert_int4(abs((x - x_min) % (x_max - x_min))) + x_min;
    int4 y2 = convert_int4(abs((x_max - x - 1) % (x_max - x_min))) + x_min;
    int4 y3 = convert_int4(abs((x - x_min) % (2 * (x_max - x_min))) + x_min) >= x_max;
    return select(normed_mod, rng_by_2 - normed_mod - (int4)(1,1,1,1), normed_mod >= rng) + x_min;
}
'''

CONVOLVE_KERNEL = '''
void __kernel convolve_kernel(
    const __global float* X, int4 X_strides, int4 X_shape, int X_offset,
    const __global float* h, int h_stride, int h_shape, int h_offset,
    __global float* Y, int4 Y_strides, int4 Y_shape, int Y_offset,
    int axis)
{
    int4 out_coord = { get_global_id(0), get_global_id(1), get_global_id(2), 0 };
    struct array_spec X_spec = { .strides = X_strides, .shape = X_shape, .offset = X_offset };
    struct array_spec Y_spec = { .strides = Y_strides, .shape = Y_shape, .offset = Y_offset };
    
    if(any(out_coord >= Y_spec.shape))
        return;

    float output = 0;
    int4 sample_coord = out_coord;

    int4 coord_min = { 0, 0, 0, 0 };
    int4 coord_max = X_spec.shape;

    for(int d=0; d<h_shape; ++d) {
        // on any sensible implementation, this switch will be optimised out being conditional on a constant
        switch(axis) {
            case 0:
                sample_coord.x = out_coord.x + ((h_shape-1)>>1) - d;
                break;
            case 1:
                sample_coord.y = out_coord.y + ((h_shape-1)>>1) - d;
                break;
            case 2:
                sample_coord.z = out_coord.z + ((h_shape-1)>>1) - d;
                break;
        }

        sample_coord = reflect(sample_coord, coord_min, coord_max);
        output += h[h_offset + d*h_stride] * X[coord_to_offset(sample_coord, X_spec)];
    }

    Y[coord_to_offset(out_coord, Y_spec)] = output;
}
'''

DFILTER_KERNEL = '''
void __kernel convolve_kernel(
    const __global float* X, int4 X_strides, int4 X_shape, int X_offset,
    const __global float* h, int h_stride, int h_shape, int h_offset,
    __global float* Y, int4 Y_strides, int4 Y_shape, int Y_offset,
    int axis)
{
    int4 global_coord = { get_global_id(0), get_global_id(1), get_global_id(2), 0 };
    struct array_spec X_spec = { .strides = X_strides, .shape = X_shape, .offset = X_offset };
    struct array_spec Y_spec = { .strides = Y_strides, .shape = Y_shape, .offset = Y_offset };

    // A vector of flags with the convolution direction set
    int4 axis_flag = (int4)(axis,axis,axis,axis) == (int4)(0,1,2,3);

    // Each run of this kernel outputs *two* pixels: the result of convolving
    // 'odd' samples (0, 2, 4, ...) with h and the result of convolving 'even'
    // samples (1, 3, 5, ... ) with reverse(h).

    // Compute the base output co-ordinate and a vector which has 1 set in the
    // component corresponding to *axis*.
    int4 output_coord = select(global_coord, global_coord * 2, axis_flag);
    int4 one_px_advance = select((int4)(0,0,0,0), (int4)(1,1,1,1), axis_flag);

    if(any(output_coord >= Y_shape))
        return;

    int4 X_coord = select(global_coord, global_coord * 4, axis_flag);

    int4 coord_min = { 0, 0, 0, 0 };
    int4 coord_max = X_spec.shape;

    float output_1 = 0, output_2 = 0;

    int m = h_shape>>1;
    for(int d=0; d<m; ++d) {
        int X_offset = 4*((m>>1)-d);

        float ha_odd = h[h_offset + (d*2)*h_stride];
        float ha_even = h[h_offset + (1+(d*2))*h_stride];

        float Xo1 = X[coord_to_offset(reflect(X_coord - (X_offset-1)*one_px_advance, coord_min, coord_max), X_spec)];
        float Xo2 = X[coord_to_offset(reflect(X_coord - (X_offset-3)*one_px_advance, coord_min, coord_max), X_spec)];
        output_1 += ha_odd * Xo1 + ha_even * Xo2;

        float Xe1 = X[coord_to_offset(reflect(X_coord + (X_offset)*one_px_advance, coord_min, coord_max), X_spec)];
        float Xe2 = X[coord_to_offset(reflect(X_coord + (X_offset+2)*one_px_advance, coord_min, coord_max), X_spec)];
        output_2 += ha_even * Xe1 + ha_odd * Xe2;
    }

    Y[coord_to_offset(output_coord, Y_spec)] = output_1;
    Y[coord_to_offset(output_coord + one_px_advance, Y_spec)] = output_2;
}
'''
