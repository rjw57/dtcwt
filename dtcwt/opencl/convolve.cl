#ifndef KERNEL_HALF_WIDTH
#   error KERNEL_HALF_WIDTH must be defined
#endif

#ifndef CHUNK_SIZE
#   error CHUNK_SIZE must be defined
#endif

#if CHUNK_SIZE < KERNEL_HALF_WIDTH
#   error CHUNK_SIZE must be at least KERNEL_HALF_WIDTH
#endif

#define LOCAL_WIDTH (CHUNK_SIZE + 2*(KERNEL_HALF_WIDTH))

typedef float scalar_t;
typedef float2 vec2_t;
typedef float4 vec4_t;

inline int2 edge_reflect(int2 coord, int2 shape)
{
    // Reflect around zero
    int2 zrx = select(coord, -coord-1, coord<0);
    int2 modx = zrx % (shape<<1);
    return min(modx, (shape<<1)-1-modx);
}

__kernel
void test_edge_reflect(
        int2 input_origin, int2 input_shape,
        __global vec2_t* output_ptr, int output_start, int2 output_strides, int2 output_shape)
{
    output_ptr += output_start;

    int2 output_coord = (int2)(get_global_id(0), get_global_id(1));
    if(any(output_coord < 0) || any(output_coord >= output_shape)) {
        return;
    }

    int2 input_coord = edge_reflect(
        input_origin + (int2)(get_global_id(0), get_global_id(1)),
        input_shape);

    vec2_t output = convert_float2(input_coord);

    int2 output_offsets = mul24(output_coord, output_strides);
    output_ptr[output_offsets.x + output_offsets.y] = output;
}

// Convolve along first dimension of data
__kernel
__attribute__((reqd_work_group_size(CHUNK_SIZE, CHUNK_SIZE, 1)))
void convolve_scalar(
    __constant vec2_t* kernel_coeffs,
    __global scalar_t* input_ptr, int input_start, int2 input_strides, int2 input_shape,
    __global vec2_t* output_ptr, int output_start, int2 output_strides, int2 output_shape)
{
    input_ptr += input_start;
    output_ptr += output_start;

    // Allocate a region of local storage to store cached input
    __local scalar_t input_cache[LOCAL_WIDTH*CHUNK_SIZE];

    // What centre input pixel does this thread correspond to?
    int2 input_coord = (int2)(get_global_id(0), get_global_id(1));

    // What pixel in the shared memory does this correspond to?
    int2 local_coord = (int2)(get_local_id(0) + KERNEL_HALF_WIDTH, get_local_id(1));

    // Read input into shared memory
    int2 input_offsets = mul24(edge_reflect(input_coord, input_shape), input_strides);
    input_cache[local_coord.x + local_coord.y*LOCAL_WIDTH] =
        input_ptr[input_offsets.x + input_offsets.y];

    // Read left apron
    if(get_local_id(0) < KERNEL_HALF_WIDTH) {
        input_offsets = mul24(
            edge_reflect(input_coord-(int2)(KERNEL_HALF_WIDTH,0), input_shape),
            input_strides
        );
        input_cache[local_coord.x + local_coord.y*LOCAL_WIDTH - KERNEL_HALF_WIDTH] =
            input_ptr[input_offsets.x + input_offsets.y];
    }

    // Read right apron
    if(get_local_id(0) >= CHUNK_SIZE - KERNEL_HALF_WIDTH) {
        input_offsets = mul24(
            edge_reflect(input_coord+(int2)(KERNEL_HALF_WIDTH,0), input_shape),
            input_strides
        );
        input_cache[local_coord.x + local_coord.y*LOCAL_WIDTH + KERNEL_HALF_WIDTH] =
            input_ptr[input_offsets.x + input_offsets.y];
    }

    // Wait for all threads to have written local memory.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform early out here after reading input since even if this thread
    // doesn't write output, it still may read useful input.
    int2 output_coord = (int2)(get_global_id(0), get_global_id(1));
    if(any(output_coord < 0) || any(output_coord >= output_shape)) {
        return;
    }

    vec2_t output = {0,0}; // input_cache[local_coord.x + local_coord.y*LOCAL_WIDTH];
    for(int k_idx=-KERNEL_HALF_WIDTH; k_idx<=KERNEL_HALF_WIDTH; ++k_idx) {
        scalar_t input_val = input_cache[k_idx + local_coord.x + local_coord.y*LOCAL_WIDTH];
        vec2_t kernel_val = kernel_coeffs[k_idx + KERNEL_HALF_WIDTH];
        output += kernel_val * input_val;
    }
    int2 output_offsets = mul24(output_coord, output_strides);
    output_ptr[output_offsets.x + output_offsets.y] = output;
}

// vim:sw=4:sts=4:et
