#ifndef CHUNK_SIZE
#   error CHUNK_SIZE must be defined
#endif

typedef float scalar_t;
typedef float2 vec2_t;
typedef float3 vec3_t;
typedef float4 vec4_t;

typedef float2 complex_t;

__kernel
void q2c(
    __global vec4_t* input_ptr, int input_start, int2 input_strides, int2 input_shape,
    __global scalar_t* low_ptr, int low_start, int2 low_strides, int2 low_shape,
    __global complex_t* high_ptr, int high_start, int3 high_strides, int3 high_shape)
{
    input_ptr += input_start;
    high_ptr += high_start;
    low_ptr += low_start;

    // Abort if we will read from invalid input
    int2 input_coord = 2 * (int2)(get_global_id(0), get_global_id(1));
    if(any(input_coord < 0) || any(input_coord >= input_shape-1)) {
        return;
    }

    // Each output pixel corresponds to a 2x2 region of the input. Read each portion.
    int2 input_offsets = mul24(input_coord, input_strides);
    int input_offset = input_offsets.x + input_offsets.y;
    vec4_t tl = input_ptr[input_offset];
    vec4_t tr = input_ptr[input_offset + input_strides.y];
    vec4_t bl = input_ptr[input_offset + input_strides.x];
    vec4_t br = input_ptr[input_offset + input_strides.x + input_strides.y];

    // Write lowpass output
    int2 low_coord = 2*(int2)(get_global_id(0), get_global_id(1));
    if(!any(low_coord < 0) && !any(low_coord >= low_shape-1)) {
        int2 low_offsets = mul24(low_coord, low_strides);
        int low_offset = low_offsets.x + low_offsets.y;
        low_ptr[low_offset] = tl.x;
        low_ptr[low_offset + low_strides.y] = tr.x;
        low_ptr[low_offset + low_strides.x] = bl.x;
        low_ptr[low_offset + low_strides.x + low_strides.y] = br.x;
    }

    // Write highpass output
    int2 high_coord = (int2)(get_global_id(0), get_global_id(1));
    if(!any(high_coord < 0) && !any(high_coord >= high_shape.xy)) {
        int2 high_offsets = mul24(high_coord.xy, high_strides.xy);
        int high_offset = high_offsets.x + high_offsets.y;

        // p = (tl + j*tr) / sqrt(2)
        // q = (br - j*bl) / sqrt(2)
        // z1 = p - q, z2 = p + q
        //
        // => z1 = ((tl-br) + j*(tr+bl)) / sqrt2
        // => z2 = ((tl+br) + j*(tr-bl)) / sqrt2

        float sqrt_half = sqrt(0.5);
        vec3_t z1_real = (tl.yzw-br.yzw) * sqrt_half;
        vec3_t z1_imag = (tr.yzw+bl.yzw) * sqrt_half;
        vec3_t z2_real = (tl.yzw+br.yzw) * sqrt_half;
        vec3_t z2_imag = (tr.yzw-bl.yzw) * sqrt_half;

        high_ptr[high_offset + 0*high_strides.z] = (complex_t)(z1_real.y, z1_imag.y);
        high_ptr[high_offset + 1*high_strides.z] = (complex_t)(z1_real.z, z1_imag.z);
        high_ptr[high_offset + 2*high_strides.z] = (complex_t)(z1_real.x, z1_imag.x);
        high_ptr[high_offset + 3*high_strides.z] = (complex_t)(z2_real.x, z2_imag.x);
        high_ptr[high_offset + 4*high_strides.z] = (complex_t)(z2_real.z, z2_imag.z);
        high_ptr[high_offset + 5*high_strides.z] = (complex_t)(z2_real.y, z2_imag.y);
    }
}

// vim:sw=4:sts=4:et
