// Required defines:

// width of filter *MUST BE ODD*
#ifndef FILTER_WIDTH
#   error "Filter width must be defined"
#endif
#if (FILTER_WIDTH & 0x1) != 1
#   error "Filter width must be odd"
#endif

// work group size as would be returned by get_local_size(0)
#ifndef LOCAL_SIZE_0
#   error "Local size 0 must be defined"
#endif
// work group size as would be returned by
// get_local_size(1) * ... * get_local_size(get_work_dim()-1)
#ifndef LOCAL_SIZE_REST
#   error "Local size 1...rest must be defined"
#endif

// datatype of input
#ifndef INPUT_TYPE
#   error "Input data type must be defined"
#endif

// Derived values
#define FILTER_HALF_WIDTH ((FILTER_WIDTH-1)>>1)
#define LOCAL_CACHE_WIDTH ((2*FILTER_HALF_WIDTH)+LOCAL_SIZE_0)

// Return a linear offset to the specified element
int index(int4 coord, int4 strides) {
    // unfortunately dot() is only defined for floating point types
    int4 prod = coord * strides;
    return prod.x + prod.y + prod.z + prod.w;
}

// Convolve along first axis. To avoid this swizzle input_{...} appropriately.
// Strides, offsets, skips and shapes are measured in units of INPUT_TYPE.
//
// The shape of the region of input pixels to process is specified by
// output_shape. *THIS IS NOT INCLUDING SKIP*. Processing a region of (N,M) in
// shape with a skip of (2,3) will result in (2N,3M) pixels of input and output
// being touched.
//
// Processing will start at offset input_offset from zero.
// Neighbouring pixels are assumed to occur input_skip pixels along each
// dimension (this would usually be 1) with each dimension requiring
// input_strides to advance. Note that, for offsets which are multiples of
// input_skip, setting input_offset = input_offset / input_skip and
// input_strides = input_strides / input_skip and then input_skip = 1 will have
// the same effect.
//
// IMPORTANT: Setting input_offset, output_offset or output_shape such that
// pixels in an invalid region are accessed is undefined and not checked for!
__kernel void convolve(
    __global INPUT_TYPE* input,
    int4 input_offset, int4 input_skip, int4 input_strides,
    __constant float* filter_kernel,
    __global INPUT_TYPE* output,
    int4 output_offset, int4 output_skip, int4 output_shape, int4 output_strides)
{
    // Create an appropriately sized region of local memory which can hold the
    // input plus some apron.
    __local INPUT_TYPE input_cache[LOCAL_CACHE_WIDTH*LOCAL_SIZE_REST];

    event_t input_copy_event;

    // Compute upper-left corner of this work group in input and output
    int4 group_coord = (int4)(
        get_group_id(0) * get_local_size(0), get_group_id(1) * get_local_size(1),
        0, 0
    );
    int4 input_origin = input_offset + input_skip * group_coord;
    int4 output_origin = output_offset + output_skip * group_coord;
    int4 local_coord = (int4)(get_local_id(0), get_local_id(1), 0, 0);

    for(int w=0; w<output_shape.w;
        ++w, ++output_origin.w, ++input_origin.w, ++local_coord.w)
    {
        input_origin.z = input_offset.z + input_skip.z * group_coord.z;
        output_origin.z = output_offset.z + output_skip.z * group_coord.z;
        local_coord.z = 0;
        for(int z=0; z<output_shape.z;
            ++z, ++output_origin.z, ++input_origin.z, ++local_coord.z)
        {
            // Copy input into cache (note that stride applies always to non-local memory)
            for(int copy_idx=0; copy_idx<LOCAL_SIZE_REST; ++copy_idx) {
                input_copy_event = async_work_group_strided_copy(
                    input_cache + LOCAL_CACHE_WIDTH * copy_idx,
                    input + index(
                        input_origin + input_skip * (int4)(-FILTER_HALF_WIDTH,copy_idx,0,0),
                        input_strides),
                    LOCAL_CACHE_WIDTH,
                    input_strides.x * input_skip.x, input_copy_event
                );
            }
            wait_group_events(1, &input_copy_event);

            // generate output pixel value
            float filter_tap;
            INPUT_TYPE output_value = 0.f, input_value;
            for(int f_idx=0; f_idx<FILTER_WIDTH; ++f_idx) {
                input_value = input_cache[
                    get_local_id(0) + f_idx +
                    get_local_id(1) * LOCAL_CACHE_WIDTH
                ];
                filter_tap = filter_kernel[f_idx];
                output_value += input_value * filter_tap;
            }

            // write output pixel value
            output[index(output_origin + output_skip*local_coord, output_strides)] = output_value;
        }
    }
}
