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

// magic function to reflect the sampling co-ordinate about the
// *outer edges* of pixel co-ordinates x_min, x_max. The output will
// always be in the range (x_min, x_max].
int4 reflect(int4 x, int4 x_min, int4 x_max)
{
    int4 rng = x_max - x_min;
    int4 rng_by_2 = 2 * rng;
    int4 mod = (x - x_min) % rng_by_2;
    int4 normed_mod = select(mod, mod + rng_by_2, mod < 0);
    return select(normed_mod, rng_by_2 - normed_mod - (int4)(1,1,1,1), normed_mod >= rng) + x_min;
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
    __constant float* filter_kernel, int4 pixels_to_write,
    __global INPUT_TYPE* input,
    int4 input_offset, int4 input_shape, int4 input_skip, int4 input_strides,
    __global INPUT_TYPE* output,
    int4 output_offset, int4 output_shape, int4 output_skip, int4 output_strides)
{
    // Create an appropriately sized region of local memory which can hold the
    // input plus some apron.
    __local INPUT_TYPE input_cache[LOCAL_CACHE_WIDTH*LOCAL_SIZE_REST];

    // Compute upper-left corner of this work group in input and output
    int4 group_coord = (int4)(
        get_group_id(0) * get_local_size(0), get_group_id(1) * get_local_size(1),
        0, 0
    );
    int4 input_origin = input_offset + input_skip * group_coord;
    int4 output_origin = output_offset + output_skip * group_coord;
    int4 local_coord = (int4)(get_local_id(0), get_local_id(1), 0, 0);

    // This is the output pixel this work item should write to
    int4 output_coord = output_origin + output_skip*local_coord;

    // This is the corresponding input pixel to read from
    int4 input_coord = input_origin + input_skip*local_coord;

    for(int w=0; w<pixels_to_write.w;
        ++w, ++output_origin.w, ++input_origin.w, ++local_coord.w)
    {
        input_origin.z = input_offset.z + input_skip.z * group_coord.z;
        output_origin.z = output_offset.z + output_skip.z * group_coord.z;
        local_coord.z = 0;
        for(int z=0; z<pixels_to_write.z;
            ++z, ++output_origin.z, ++input_origin.z, ++local_coord.z)
        {
            // Copy input into cache
            input_cache[get_local_id(0) + FILTER_HALF_WIDTH +
                LOCAL_CACHE_WIDTH * get_local_id(1)] = input[
                    index(clamp(input_coord, 0, input_shape-1), input_strides)];
            if(get_local_id(0) < FILTER_HALF_WIDTH) {
                input_cache[get_local_id(0) +
                    LOCAL_CACHE_WIDTH * get_local_id(1)] = input[index(
                        clamp(input_coord - input_skip*(int4)(FILTER_HALF_WIDTH,0,0,0),
                            0, input_shape-1),
                        input_strides)];
            }
            if(get_local_id(0) >= get_local_size(0) - FILTER_HALF_WIDTH) {
                input_cache[get_local_id(0) + 2*(FILTER_HALF_WIDTH) +
                    LOCAL_CACHE_WIDTH * get_local_id(1)] = input[index(
                        clamp(input_coord + input_skip*(int4)(FILTER_HALF_WIDTH,0,0,0),
                            0, input_shape-1),
                        input_strides)];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // abort if we're writing outside the valid region. Do so now
            // because we may still have read something important into the
            // input cache.
            if(any(output_coord < 0) || any(output_coord > output_shape)) {
                continue;
            }

            // generate output pixel value
            float filter_tap;
            INPUT_TYPE output_value = 0.f, input_value;
            for(int f_idx=0; f_idx<FILTER_WIDTH; ++f_idx) {
                input_value = input_cache[
                    get_local_id(0) + f_idx +
                    get_local_id(1) * LOCAL_CACHE_WIDTH
                ];
                //input_value = 1.f;
                filter_tap = filter_kernel[f_idx];
                output_value += input_value * filter_tap;
            }

            // write output pixel value
            output[index(output_coord, output_strides)] = output_value;
        }
    }
}
