#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <benchmark.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <layer_type.h>
#include <net.h>


unsigned char model_density_bitfield[262144] = { 0 };
float model_pos_encoder_hash_table[11420064] = { 0 };

float xyz_encoder_weight_0_32x64[32 * 64] = { 0 };
float xyz_encoder_weight_1_64x16[64 * 16] = { 0 };



static const char clean_float_kernel[] = R"(
#version 450

layout (binding = 0) buffer io { float io_data[]; };

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    io_data[gx] = float(0.0f);
}
)";


static const char clean_int_kernel[] = R"(
#version 450

layout (binding = 0) buffer io { int io_data[]; };

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    io_data[gx] = int(0);
}
)";


static const char ray_aabb_intersect_kernel[] = R"(
#version 450

layout (constant_id = 0) const float NEAR_DISTANCE = 0.01;

layout (binding = 0) readonly buffer rays_o { float rays_o_data[]; };
layout (binding = 1) readonly buffer rays_d { float rays_d_data[]; };
layout (binding = 2) writeonly buffer hits_t { float hits_t_data[]; };

layout (push_constant) uniform parameter
{
    float center;
    float half_size;
} p;

void main()
{
    int gx = int(gl_GlobalInvocationID.x);

    int input_offset = 3 * gx;
    int output_offset = 2 * gx;

    float ray_o_0 = rays_o_data[0];
    float ray_o_1 = rays_o_data[1];
    float ray_o_2 = rays_o_data[2];

    float inv_d_0 = 1.0 / rays_d_data[input_offset + 0];
    float inv_d_1 = 1.0 / rays_d_data[input_offset + 1];
    float inv_d_2 = 1.0 / rays_d_data[input_offset + 2];

    float t_min_0 = (p.center - p.half_size - ray_o_0) * inv_d_0;
    float t_min_1 = (p.center - p.half_size - ray_o_1) * inv_d_1;
    float t_min_2 = (p.center - p.half_size - ray_o_2) * inv_d_2;

    float t_max_0 = (p.center + p.half_size - ray_o_0) * inv_d_0;
    float t_max_1 = (p.center + p.half_size - ray_o_1) * inv_d_1;
    float t_max_2 = (p.center + p.half_size - ray_o_2) * inv_d_2;

    float _t1_0 = min(t_min_0, t_max_0);
    float _t1_1 = min(t_min_1, t_max_1);
    float _t1_2 = min(t_min_2, t_max_2);

    float _t2_0 = max(t_min_0, t_max_0);
    float _t2_1 = max(t_min_1, t_max_1);
    float _t2_2 = max(t_min_2, t_max_2);

    float t1 = max(max(_t1_0,_t1_1),_t1_2);
    float t2 = min(min(_t2_0,_t2_1),_t2_2);

    if (t2 > 0) {
        hits_t_data[output_offset + 0] = max(t1, NEAR_DISTANCE);
        hits_t_data[output_offset + 1] = t2;
    }
    else {
        hits_t_data[output_offset + 0] = -1.0;
        hits_t_data[output_offset + 1] = -1.0;
    }
}
)";


static const char raymarching_test_kernel[] = R"(
#version 450

layout (constant_id = 0) const float SQRT3_MAX_SAMPLES = 0;
layout (constant_id = 1) const float SQRT3_2 = 0;

layout (binding = 0) readonly buffer rays_o { float rays_o_data[]; };
layout (binding = 1) readonly buffer rays_d { float rays_d_data[]; };
layout (binding = 2) buffer hits_t { float hits_t_data[]; };
layout (binding = 3) readonly buffer alive_indices { int alive_indices_data[]; };
layout (binding = 4) readonly buffer density_bitfield { int density_bitfield_data[]; };

layout (binding = 5) writeonly buffer ray_indices { int ray_indices_data[]; };
layout (binding = 6) writeonly buffer valid_mask { int valid_mask_data[]; };
layout (binding = 7) writeonly buffer deltas { float deltas_data[]; };
layout (binding = 8) writeonly buffer ts { float ts_data[]; };
layout (binding = 9) writeonly buffer samples_counter { int samples_counter_data[]; };

layout (push_constant) uniform parameter
{
    int cascades;
    int grid_size;
    float scale;
    float exp_step_factor;
    int max_samples;
} p;

float calc_dt(float t, float exp_step_factor, int grid_size, float scale)
{
    return clamp(t * exp_step_factor, SQRT3_MAX_SAMPLES, SQRT3_2 * scale / float(grid_size));
}

int frexp_bit(float x) {
    int exponent = 0;
    if (x != 0.0) {
        uint bits = floatBitsToUint(x);
        exponent = int((bits & uint(0x7f800000)) >> 23) - 127;
        bits &= uint(0x7fffff);
        bits |= uint(0x3f800000);
        float frac = uintBitsToFloat(bits);
        if (frac < 0.5) {
            exponent -= 1;
        } else if (frac > 1.0) {
            exponent += 1;
        }
    }
    return exponent;
}

int mip_from_pos(float xyz_0, float xyz_1, float xyz_2, int cascades)
{
    float mx = max(max(abs(xyz_0),abs(xyz_1)),abs(xyz_2));
    int exponent = frexp_bit(mx) + 1;
    return min(cascades - 1, max(0, exponent));
}

int mip_from_dt(float dt, int grid_size, int cascades)
{
    int exponent = frexp_bit(float(dt * grid_size));
    return min(cascades - 1, max(0, exponent));
}

uint expand_bits(uint v) {
    v = (v * uint(0x00010001)) & uint(0xFF0000FF);
    v = (v * uint(0x00000101)) & uint(0x0F00F00F);
    v = (v * uint(0x00000011)) & uint(0xC30C30C3);
    v = (v * uint(0x00000005)) & uint(0x49249249);
    return v;
}

uint morton3D(uint x, uint y, uint z) {
    uvec3 xyz = uvec3(x, y, z);
    xyz = uvec3(expand_bits(xyz.x), expand_bits(xyz.y), expand_bits(xyz.z));
    return xyz.x | (xyz.y << 1u) | (xyz.z << 2u);
}

void main()
{
    int n = int(gl_GlobalInvocationID.x);

    int r = alive_indices_data[n];
    int grid_size3 = p.grid_size * p.grid_size * p.grid_size;
    float grid_size_inv = 1.0 / p.grid_size;
    
    float ray_o_0 = rays_o_data[0];
    float ray_o_1 = rays_o_data[1];
    float ray_o_2 = rays_o_data[2];

    float ray_d_0 = rays_d_data[3 * r + 0];
    float ray_d_1 = rays_d_data[3 * r + 1];
    float ray_d_2 = rays_d_data[3 * r + 2];

    float d_inv_0 = 1.0 / ray_d_0;
    float d_inv_1 = 1.0 / ray_d_1;
    float d_inv_2 = 1.0 / ray_d_2;

    float t = hits_t_data[2 * r + 0];
    float t2 = hits_t_data[2 * r + 1];

    int s = 0;
    int start_based = n * p.max_samples;

    while ( (0<t) && (t<t2) && (s<p.max_samples) ) {
        float xyz_0 = ray_o_0 + t * ray_d_0;
        float xyz_1 = ray_o_1 + t * ray_d_1;
        float xyz_2 = ray_o_2 + t * ray_d_2;

        float dt = calc_dt(t, p.exp_step_factor, p.grid_size, p.scale);
        int mip = max(mip_from_pos(xyz_0,xyz_1,xyz_2,p.cascades),mip_from_dt(dt,p.grid_size,p.cascades));

        float mip_bound = min(pow(2., mip - 1), p.scale);
        float mip_bound_inv = 1.0 / mip_bound;

        float nxyz_0 = clamp(0.5 * (xyz_0 * mip_bound_inv + 1) * p.grid_size, 0.0, p.grid_size - 1.0);
        float nxyz_1 = clamp(0.5 * (xyz_1 * mip_bound_inv + 1) * p.grid_size, 0.0, p.grid_size - 1.0);
        float nxyz_2 = clamp(0.5 * (xyz_2 * mip_bound_inv + 1) * p.grid_size, 0.0, p.grid_size - 1.0);

        int idx = mip * grid_size3 + int(morton3D(uint(nxyz_0),uint(nxyz_1),uint(nxyz_2)));
        uint occ = density_bitfield_data[idx/8u] & (1u << (idx % 8u));

        if (occ > 0) {
            idx = start_based + s;
            ray_indices_data[idx] = r;
            valid_mask_data[idx] = 1;
            ts_data[idx] = t;
            deltas_data[idx] = dt;
            t += dt;
            hits_t_data[2 * r + 0] = t;
            s += 1;
        }
        else {
            float txyz_0 = (((nxyz_0 + 0.5 + 0.5 * sign(ray_d_0)) * grid_size_inv * 2 - 1.0) * mip_bound - xyz_0) * d_inv_0;
            float txyz_1 = (((nxyz_1 + 0.5 + 0.5 * sign(ray_d_1)) * grid_size_inv * 2 - 1.0) * mip_bound - xyz_1) * d_inv_1;
            float txyz_2 = (((nxyz_2 + 0.5 + 0.5 * sign(ray_d_2)) * grid_size_inv * 2 - 1.0) * mip_bound - xyz_2) * d_inv_2;

            float t_target = t + max(0, min(min(txyz_0,txyz_1),txyz_2));
            t += calc_dt(t, p.exp_step_factor, p.grid_size, p.scale);
            while (t < t_target) {
                t += calc_dt(t, p.exp_step_factor, p.grid_size, p.scale);
            }
        }
    }

    samples_counter_data[n] = s;
}
)";


static const char hash_encoder_kernel[] = R"(
#version 450

layout (constant_id = 0) const float log_per_level_scale = 0.2772588722239781;
layout (constant_id = 1) const int base_res = 16;
layout (constant_id = 2) const int feat_dim = 2;
layout (constant_id = 3) const int begin_fast_hash_level = 6;

layout (binding = 0) readonly buffer xyzs { float xyzs_data[]; };
layout (binding = 1) readonly buffer table { float table_data[]; };
layout (binding = 2) writeonly buffer output_embedding { float output_embedding_data[]; };
layout (binding = 3) readonly buffer hash_map_sizes { int hash_map_sizes_data[]; };
layout (binding = 4) readonly buffer offsets { int offsets_data[]; };


layout (push_constant) uniform parameter
{
    int B;
    int out_dim;
} p;

float grid_scale(int level, float log_scale, int base_res) {
    float exp_scale = exp(float(level) * log_scale);
    return float(base_res) * exp_scale - 1.0;
}

uint grid_resolution(float scale) {
    return uint(ceil(scale)) + 1u;
}

uint fast_hash(uvec3 pos_grid_local) {
    uint result = uint(0);
    uvec3 primes = uvec3(uint(1), uint(2654435761), uint(805459861));
    for (int i = 0; i < 3; ++i) {
        result ^= uint(pos_grid_local[i]) * primes[i];
    }
    return result;
}

uint under_hash(uvec3 pos_grid_local, uint resolution) {
    uint result = uint(0);
    uint stride = uint(1);
    for (int i = 0; i < 3; ++i) {
        result += uint(pos_grid_local[i] * stride);
        stride *= resolution;
    }
    return result;
}

uint grid_pos2hash_index(uint indicator, uvec3 pos_grid_local, uint resolution, uint map_size) {
    uint hash_result = uint(0);
    if (indicator > 0) {
        hash_result = under_hash(pos_grid_local, resolution);
    }
    else {
        hash_result = fast_hash(pos_grid_local);
    }
    return hash_result % map_size;
}

void main()
{
    int i = int(gl_GlobalInvocationID.x); // B的维度
    int level = int(gl_GlobalInvocationID.y); // hash_level的维度

    vec3 xyz = vec3(xyzs_data[3 * i + 0], xyzs_data[3 * i + 1], xyzs_data[3 * i + 2]);

    float scale = grid_scale(level, log_per_level_scale, base_res);
    uint resolution =  grid_resolution(scale);

    int offset = offsets_data[level] * feat_dim;

    vec3 pos = xyz * vec3(scale) + vec3(0.5);
    uvec3 pos_grid = uvec3(floor(pos));
    pos -= vec3(pos_grid);

    int map_size = hash_map_sizes_data[level];

    vec2 local_features = vec2(0.0f, 0.0f);

    for (int idx = 0; idx < 8; idx++) {
        float w = 1.0f;
        uvec3 pos_grid_local = uvec3(0, 0, 0);

        for (int d = 0; d < 3; d++) {
            if ((idx & (1u << d)) == 0u) {
                pos_grid_local[d] = pos_grid[d];
                w *= 1 - pos[d];
            }
            else {
                pos_grid_local[d] = pos_grid[d] + 1u;
                w *= pos[d];
            }
        }

        uint index = grid_pos2hash_index(uint(level < begin_fast_hash_level), pos_grid_local, resolution, map_size);
        int index_table = int(offset + index * feat_dim);

        for (int l_f = 0; l_f < feat_dim; l_f++) {
            local_features[l_f] += w * table_data[index_table + l_f];
        }
    }

    int out_index_base = level * feat_dim ;
    for (int l_f = 0; l_f < feat_dim; l_f++) {
        output_embedding_data[i * p.out_dim + out_index_base + l_f] = local_features[l_f];
    }
}
)";


static const char dir_encoder_kernel[] = R"(
#version 450

layout (binding = 0) readonly buffer dirs { float dirs_data[]; };
layout (binding = 1) writeonly buffer embedding { float embedding_data[]; };

void main()
{
    int i = int(gl_GlobalInvocationID.x); // B的维度

    float x = (dirs_data[3 * i + 0] + 1.0) / 2.0;
    float y = (dirs_data[3 * i + 1] + 1.0) / 2.0;
    float z = (dirs_data[3 * i + 2] + 1.0) / 2.0;

    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    embedding_data[16 * i +  0] = (0.28209479177387814);
    embedding_data[16 * i +  1] = (-0.48860251190291987 * y);
    embedding_data[16 * i +  2] = (0.48860251190291987 * z);
    embedding_data[16 * i +  3] = (-0.48860251190291987 * x);
    embedding_data[16 * i +  4] = (1.0925484305920792 * xy);
    embedding_data[16 * i +  5] = (-1.0925484305920792 * yz);
    embedding_data[16 * i +  6] = (0.94617469575755997 * z2 - 0.31539156525251999);
    embedding_data[16 * i +  7] = (-1.0925484305920792 * xz);
    embedding_data[16 * i +  8] = (0.54627421529603959 * x2 - 0.54627421529603959 * y2);
    embedding_data[16 * i +  9] = (0.59004358992664352 * y * (-3.0 * x2 + y2));
    embedding_data[16 * i + 10] = (2.8906114426405538 * xy * z);
    embedding_data[16 * i + 11] = (0.45704579946446572 * y * (1.0 - 5.0 * z2));
    embedding_data[16 * i + 12] = (0.3731763325901154 * z * (5.0 * z2 - 3.0));
    embedding_data[16 * i + 13] = (0.45704579946446572 * x * (1.0 - 5.0 * z2));
    embedding_data[16 * i + 14] = (1.4453057213202769 * z * (x2 - y2));
    embedding_data[16 * i + 15] = (0.59004358992664352 * x * (-x2 + 3.0 * y2));
}
)";


static const char composite_test_kernel[] = R"(
#version 450

layout (binding = 0) readonly buffer sigmas { float sigmas_data[]; };
layout (binding = 1) readonly buffer rgbs { float rgbs_data[]; };
layout (binding = 2) readonly buffer deltas { float deltas_data[]; };
layout (binding = 3) readonly buffer ts { float ts_data[]; };
layout (binding = 4) readonly buffer pack_info { int pack_info_data[]; };
layout (binding = 5) buffer alive_indices { int alive_indices_data[]; };
layout (binding = 6) buffer opacity { float opacity_data[]; };
layout (binding = 7) writeonly buffer depth { float depth_data[]; };
layout (binding = 8) writeonly buffer rgb { float rgb_data[]; };

layout (push_constant) uniform parameter
{
    float T_threshold;
} p;

void main()
{
    int n = int(gl_GlobalInvocationID.x);

    int start_idx = pack_info_data[2 * n + 0];
    int steps = pack_info_data[2 * n + 1];
    int ray_idx = alive_indices_data[n];

    if (steps == 0) {
        alive_indices_data[n] = -1;
    }
    else {
        float T = 1.0 - opacity_data[ray_idx];

        vec3 rgb_temp = vec3(0.0);
        float depth_temp = 0.0;
        float opacity_temp = 0.0;

        for (int s = 0; s < steps; s++) {
            int s_n = start_idx + s;
            float delta = deltas_data[s_n];
            float a = 1.0 - exp(-1.0 * sigmas_data[s_n] * delta);

            float w = a * T;
            float tmid = ts_data[s_n];
            vec3 rgbs_vec3 = vec3(rgbs_data[3 * s_n + 0], rgbs_data[3 * s_n + 1], rgbs_data[3 * s_n + 2]);
            rgb_temp += vec3(w) * rgbs_vec3;
            depth_temp += w * tmid;
            opacity_temp += w;
            T *= 1.0 - a;

            if (T <= p.T_threshold) {
                alive_indices_data[n] = -2;
                break;
            }
        }
        
        rgb_data[3 * ray_idx + 0] += rgb_temp[0];
        rgb_data[3 * ray_idx + 1] += rgb_temp[1];
        rgb_data[3 * ray_idx + 2] += rgb_temp[2];
        depth_data[ray_idx] += depth_temp;
        opacity_data[ray_idx] += opacity_temp;
    }

}
)";


static const char gemm2_kernel[] = R"(
#version 450

// M是动态的，N和K在编译时可以确定
layout (constant_id = 0) const int N = 2048;
layout (constant_id = 1) const int K = 2048;
// 是否用Relu
layout (constant_id = 2) const int relu = 1;

layout (binding = 0) readonly  buffer A { float A_data[]; };
layout (binding = 1) readonly  buffer B { float B_data[]; };
layout (binding = 2) writeonly buffer C { float C_data[]; };

void main()
{
    const int k = int(gl_GlobalInvocationID.x);
    const int m = int(gl_GlobalInvocationID.y);

    float sum = 0.0f;
    for (uint i = 0; i < N; i++) {
        sum += (10.0 * A_data[m * N + i]) * (10.0 * B_data[i * K + k]);
    }

    if (relu > 0 && sum < 0) {
        C_data[m * K + k] = float(0);
    }
    else {
        C_data[m * K + k] = float(sum / 100.0);
    }
}
)";


ncnn::Mat get_ray_directions(int H, int W, ncnn::Mat K)
{
    ncnn::Mat directions(W, H, 3);

    float fx = K[0], fy = K[4], cx = K[2], cy = K[5];

    float* u_ptr = (float*)directions.channel(0);
    float* v_ptr = (float*)directions.channel(1);
    float* o_ptr = (float*)directions.channel(2);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            *u_ptr = (x - cx + 0.5) / fx;
            *v_ptr = (y - cy + 0.5) / fy;
            *o_ptr = 1;
            u_ptr++;
            v_ptr++;
            o_ptr++;
        }
    }

    return directions.reshape(W * H, 3);
}


int main()
{
    // 一些参数
    const int POSE = 200;
    float w = 800;
    float h = 800;
    const float downsample = 1.0;
    const int WH = 640000;
    const std::string dataset = "Ship";
    const float model_scale = 0.5;
    const float NEAR_DISTANCE = 0.01;
    const int MAX_SAMPLES = 1024;
    const float SQRT3 = 1.7320508075688772;
    const float SQRT3_MAX_SAMPLES = SQRT3 / 1024;
    const float SQRT3_2 = 1.7320508075688772 * 2;
    const int hash_encoder_hash_level = 16;
    const int hash_encoder_base_res = 16;
    const int hash_encoder_max_res = 1024;
    const int hash_encoder_feat_dim = 2; // 写死了，要改的话就要改shader代码
    const int begin_fast_hash_level = 6;
    const int hash_encoder_out_dim = 32;
    const int hash_encoder_hash_map_sizes[16] = { 4096, 10648, 21952, 50656, 117656, 262144, 524288, 524288, 524288, 524288, 524288, 524288, 524288, 524288, 524288, 524288 };
    const int hash_encoder_offsets[16] = { 0, 4096, 14744, 36696, 87352, 205008, 467152, 991440, 1515728, 2040016, 2564304, 3088592, 3612880, 4137168, 4661456, 5185744 };




    {
        std::ifstream file("assets/" + dataset + "/model_density_bitfield.dat", std::ios::in | std::ios::binary);
        file.read((char*)&model_density_bitfield, sizeof model_density_bitfield);
        file.close();
    }
    {
        std::ifstream file("assets/" + dataset + "/model_pos_encoder_hash_table.dat", std::ios::in | std::ios::binary);
        file.read((char*)&model_pos_encoder_hash_table, sizeof model_pos_encoder_hash_table);
        file.close();
    }

    //{
    //    std::ifstream file("assets/xyz_encoder_weight_0_32x64.dat", std::ios::in | std::ios::binary);
    //    file.read((char*)&xyz_encoder_weight_0_32x64, sizeof xyz_encoder_weight_0_32x64);
    //    file.close();
    //}
    //{
    //    std::ifstream file("assets/xyz_encoder_weight_1_64x16.dat", std::ios::in | std::ios::binary);
    //    file.read((char*)&xyz_encoder_weight_1_64x16, sizeof xyz_encoder_weight_1_64x16);
    //    file.close();
    //}


    // read_intrinsics
    ncnn::Mat K(3, 3);
    {
        std::ifstream in("assets/" + dataset + "/intrinsics.txt");
        float fx, fy;
        in >> fx;
        fy = fx;
        in.close();

        fx = fx * downsample;
        fy = fy * downsample;
        w = w * downsample;
        h = h * downsample;
        
        float* k_ptr = (float*)K.data;
        k_ptr[0] = fx; k_ptr[1] = 0; k_ptr[2] = w / 2;
        k_ptr[3] = 0; k_ptr[4] = fy; k_ptr[5] = h / 2;
        k_ptr[6] = 0; k_ptr[7] = 0; k_ptr[8] = 1;
    }

    int img_w = w, img_h = h;


    ncnn::Mat directions = get_ray_directions(img_h, img_w, K);


    // read_meta
    ncnn::Mat c2w_012s(3, 3, POSE, (size_t)4u, 1);
    ncnn::Mat c2w_3s(3, 1, POSE, (size_t)4u, 1);
    {
        std::ifstream in("assets/" + dataset + "/bbox.txt");
        float x_min, y_min, z_min, x_max, y_max, z_max;
        in >> x_min; in >> y_min; in >> z_min;
        in >> x_max; in >> y_max; in >> z_max;
        in.close();

        float x_shift = (x_max + x_min) / 2;
        float y_shift = (y_max + y_min) / 2;
        float z_shift = (z_max + z_min) / 2;

        float data_scale = (std::max)(x_max - x_min, (std::max)(y_max - y_min, z_max - z_min)) / 2 * 1.05;
        if (dataset == "Lego") {
            data_scale *= 1.1;
        }
        else if (dataset == "Mic") {
            data_scale *= 1.2;
        }

        {
            for (int i = 0; i < POSE; ++i) {
                std::stringstream ss;
                ss << std::setw(4) << std::setfill('0') << i;
                std::string pose_filename = "assets/" + dataset + "/pose/2_" + ss.str() + ".txt";
                // std::cout << "loading pose: " << pose_filename << std::endl;

                ncnn::Mat c2w_012 = c2w_012s.channel(i);
                ncnn::Mat c2w_3 = c2w_3s.channel(i);

                std::ifstream in(pose_filename);
                float* r_ptr = (float*)c2w_012.data;
                float* s_ptr = (float*)c2w_3.data;
                float tmp;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        in >> tmp;
                        r_ptr[i * 3 + j] = tmp;
                    }
                    in >> tmp;
                    s_ptr[i] = tmp;
                }
                in.close();

                s_ptr = (float*)c2w_3.data;
                *s_ptr = (*s_ptr - x_shift) / (2 * data_scale); s_ptr++;
                *s_ptr = (*s_ptr - y_shift) / (2 * data_scale); s_ptr++;
                *s_ptr = (*s_ptr - z_shift) / (2 * data_scale); s_ptr++;
            }
        }
    }


    
    
    // 开始vulkan旅途
    {
        ncnn::VulkanDevice* vkdev;
        
        // 初始化gpu
        ncnn::create_gpu_instance();
        vkdev = ncnn::get_gpu_device(0);
        ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
        ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
        ncnn::Option opt;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;
        opt.use_packing_layout = false;
        opt.use_shader_pack8 = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = false;
        opt.use_int8_storage = false;
        opt.use_int8_packed = false;
        opt.use_int8_arithmetic = false;
        opt.use_int8_inference = false;
        ncnn::VkCompute cmd(vkdev);

        // 初始化gemm
        ncnn::Layer* gemm;
        gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 1); // transA
        pd.set(3, 1); // transB
        pd.set(12, 1); // output elempack
        gemm->load_param(pd);
        gemm->load_model(ncnn::ModelBinFromMatArray(0));
        gemm->create_pipeline(opt);

        // 初始化clean_float_pipeline
        ncnn::Pipeline* clean_float_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(0);
            ncnn::compile_spirv_module(clean_float_kernel, sizeof(clean_float_kernel) - 1, opt, spirv);
            clean_float_pipeline = new ncnn::Pipeline(vkdev);
            clean_float_pipeline->set_optimal_local_size_xyz(32, 1, 1);
            clean_float_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化clean_int_pipeline
        ncnn::Pipeline* clean_int_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(0);
            ncnn::compile_spirv_module(clean_int_kernel, sizeof(clean_int_kernel) - 1, opt, spirv);
            clean_int_pipeline = new ncnn::Pipeline(vkdev);
            clean_int_pipeline->set_optimal_local_size_xyz(32, 1, 1);
            clean_int_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化ray_aabb_intersect_pipeline
        ncnn::Pipeline* ray_aabb_intersect_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(1);
            specializations[0].f = NEAR_DISTANCE;
            ncnn::compile_spirv_module(ray_aabb_intersect_kernel, sizeof(ray_aabb_intersect_kernel) - 1, opt, spirv);
            ray_aabb_intersect_pipeline = new ncnn::Pipeline(vkdev);
            ray_aabb_intersect_pipeline->set_optimal_local_size_xyz(32, 1, 1);
            ray_aabb_intersect_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化raymarching_test_pipeline
        ncnn::Pipeline* raymarching_test_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(2);
            specializations[0].f = SQRT3_MAX_SAMPLES;
            specializations[1].f = SQRT3_2;
            ncnn::compile_spirv_module(raymarching_test_kernel, sizeof(raymarching_test_kernel) - 1, opt, spirv);
            raymarching_test_pipeline = new ncnn::Pipeline(vkdev);
            raymarching_test_pipeline->set_optimal_local_size_xyz(32, 1, 1);
            raymarching_test_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化raymarching_test_pipeline
        ncnn::Pipeline* hash_encoder_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(4);
            specializations[0].f = log(float(hash_encoder_max_res) / float(hash_encoder_base_res)) / float(hash_encoder_hash_level - 1);
            specializations[1].i = hash_encoder_base_res;
            specializations[2].i = hash_encoder_feat_dim;
            specializations[3].i = begin_fast_hash_level;
            ncnn::compile_spirv_module(hash_encoder_kernel, sizeof(hash_encoder_kernel) - 1, opt, spirv);
            hash_encoder_pipeline = new ncnn::Pipeline(vkdev);
            hash_encoder_pipeline->set_optimal_local_size_xyz(8, 8, 1);
            hash_encoder_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化dir_encoder_pipeline
        ncnn::Pipeline* dir_encoder_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(0);
            ncnn::compile_spirv_module(dir_encoder_kernel, sizeof(dir_encoder_kernel) - 1, opt, spirv);
            dir_encoder_pipeline = new ncnn::Pipeline(vkdev);
            dir_encoder_pipeline->set_optimal_local_size_xyz(32, 1, 1);
            dir_encoder_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化composite_test_pipeline
        ncnn::Pipeline* composite_test_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(0);
            ncnn::compile_spirv_module(composite_test_kernel, sizeof(composite_test_kernel) - 1, opt, spirv);
            composite_test_pipeline = new ncnn::Pipeline(vkdev);
            composite_test_pipeline->set_optimal_local_size_xyz(32, 1, 1);
            composite_test_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化xyz_encoder_weight_0_gemm_pipeline
        ncnn::Pipeline* xyz_encoder_weight_0_gemm_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(3);
            specializations[0].i = 32; // N=32
            specializations[1].i = 64; // K=64
            specializations[2].i = 1; // with relu
            ncnn::compile_spirv_module(gemm2_kernel, sizeof(gemm2_kernel) - 1, opt, spirv);
            xyz_encoder_weight_0_gemm_pipeline = new ncnn::Pipeline(vkdev);
            xyz_encoder_weight_0_gemm_pipeline->set_optimal_local_size_xyz(8, 8, 1);
            xyz_encoder_weight_0_gemm_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化xyz_encoder_weight_1_gemm_pipeline
        ncnn::Pipeline* xyz_encoder_weight_1_gemm_pipeline;
        {
            static std::vector<uint32_t> spirv;
            std::vector<ncnn::vk_specialization_type> specializations(3);
            specializations[0].i = 64; // N=64
            specializations[1].i = 16; // K=16
            specializations[2].i = 0; // without relu
            ncnn::compile_spirv_module(gemm2_kernel, sizeof(gemm2_kernel) - 1, opt, spirv);
            xyz_encoder_weight_1_gemm_pipeline = new ncnn::Pipeline(vkdev);
            xyz_encoder_weight_1_gemm_pipeline->set_optimal_local_size_xyz(8, 8, 1);
            xyz_encoder_weight_1_gemm_pipeline->create(spirv.data(), spirv.size() * 4, specializations);
        }

        // 初始化xyz_encoder
        ncnn::Net xyz_encoder_net;
        //xyz_encoder_net.opt.use_vulkan_compute = true;
        //xyz_encoder_net.opt.blob_vkallocator = blob_vkallocator;
        //xyz_encoder_net.opt.workspace_vkallocator = blob_vkallocator;
        //xyz_encoder_net.opt.staging_vkallocator = staging_vkallocator;
        //xyz_encoder_net.opt.use_fp16_arithmetic = false;
        //xyz_encoder_net.opt.use_fp16_storage = false;
        //xyz_encoder_net.opt.use_fp16_packed = false;
        xyz_encoder_net.load_param(("assets/" + dataset + "/xyz_encoder.param").c_str());
        xyz_encoder_net.load_model(("assets/" + dataset + "/xyz_encoder.bin").c_str());

        // 初始化rgb_net
        ncnn::Net rgb_net;
        //rgb_net.opt.use_vulkan_compute = true;
        //rgb_net.opt.blob_vkallocator = blob_vkallocator;
        //rgb_net.opt.workspace_vkallocator = blob_vkallocator;
        //rgb_net.opt.staging_vkallocator = staging_vkallocator;
        //rgb_net.opt.use_fp16_arithmetic = false;
        //rgb_net.opt.use_fp16_storage = false;
        //rgb_net.opt.use_fp16_packed = false;
        rgb_net.load_param(("assets/" + dataset + "/rgb_net.param").c_str());
        rgb_net.load_model(("assets/" + dataset + "/rgb_net.bin").c_str());

        // 预上传hash_table
        ncnn::VkMat params_gpu;
        {
            ncnn::Mat params(11420064);
            float* p = (float*)params.data;
            for (int i = 0; i < 11420064; i++) {
                p[i] = model_pos_encoder_hash_table[i];
            }
            cmd.record_clone(params, params_gpu, opt);
            cmd.submit_and_wait();
            cmd.reset();
        }

        //// 预上传xyz_encoder_weight_0_gpu
        //ncnn::VkMat xyz_encoder_weight_0_gpu;
        //{
        //    ncnn::Mat xyz_encoder_weight_0(32 * 64);
        //    float* p = (float*)xyz_encoder_weight_0.data;
        //    for (int i = 0; i < 32 * 64; i++) {
        //        p[i] = xyz_encoder_weight_0_32x64[i];
        //    }
        //    cmd.record_clone(xyz_encoder_weight_0, xyz_encoder_weight_0_gpu, opt);
        //    cmd.submit_and_wait();
        //    cmd.reset();
        //}

        //// 预上传xyz_encoder_weight_1_gpu
        //ncnn::VkMat xyz_encoder_weight_1_gpu;
        //{
        //    ncnn::Mat xyz_encoder_weight_1(64 * 16);
        //    float* p = (float*)xyz_encoder_weight_1.data;
        //    for (int i = 0; i < 64 * 16; i++) {
        //        p[i] = xyz_encoder_weight_1_64x16[i];
        //    }
        //    cmd.record_clone(xyz_encoder_weight_1, xyz_encoder_weight_1_gpu, opt);
        //    cmd.submit_and_wait();
        //    cmd.reset();
        //}


        // 循环生成每一个pose的图像
        for (int pose_idx = 0; pose_idx < POSE; pose_idx++) {


            ncnn::Mat c2w_012 = c2w_012s.channel(pose_idx);
            ncnn::Mat c2w_3 = c2w_3s.channel(pose_idx);


            double start_time = ncnn::get_current_time();



            /////////////////////////////////////////////////////////////////////////////// get_rays

            // 上传
            ncnn::VkMat input0_gpu, input1_gpu;
            cmd.record_clone(directions, input0_gpu, opt);
            cmd.record_clone(c2w_012, input1_gpu, opt);

            // forward
            std::vector<ncnn::VkMat> inputs_gpu(2);
            inputs_gpu[0] = input0_gpu;
            inputs_gpu[1] = input1_gpu;
            std::vector<ncnn::VkMat> outputs_gpu(1);
            gemm->forward(inputs_gpu, outputs_gpu, cmd, opt);
            cmd.submit_and_wait();
            cmd.reset();

            ncnn::Mat rays_o = c2w_3; // (3)
            ncnn::VkMat rays_d_gpu = outputs_gpu[0]; // (640000,3)


            /////////////////////////////////////////////////////////////////////////////// ray_aabb_intersection
            ncnn::VkMat hits_t_gpu;
            ncnn::VkMat rays_o_gpu;
            {
                cmd.record_clone(c2w_3, rays_o_gpu, opt);
                //hits_t_gpu.create(2, 640000, 4u, 1, blob_vkallocator);
                {
                    ncnn::Mat hits_t_cpu(2, WH); hits_t_cpu.fill<float>(0.0f);
                    cmd.record_clone(hits_t_cpu, hits_t_gpu, opt);
                }

                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = rays_o_gpu;
                bindings[1] = rays_d_gpu;
                bindings[2] = hits_t_gpu;

                std::vector<ncnn::vk_constant_type> constants(2);
                constants[0].f = 0;
                constants[1].f = 0.5;

                ncnn::VkMat dispatcher;
                dispatcher.w = WH; // 640000
                dispatcher.h = 1;
                dispatcher.c = 1;

                cmd.record_pipeline(ray_aabb_intersect_pipeline, bindings, constants, dispatcher);

                cmd.submit_and_wait();
                cmd.reset();
            }


            /////////////////////////////////////////////////////////////////////////////// __render_rays_test
            float exp_step_factor = 0;
            if (model_scale > 0.5)
                exp_step_factor = 1 / 256;
            float T_threshold = 1e-4;
            int max_samples = MAX_SAMPLES;

            int N_rays = WH; // 光线的数量 // len(rays_o)
            int samples = 0, total_samples = 0;
            int min_samples = 4;
            if (exp_step_factor == 0)
                min_samples = 1;

            // 初始的有效下标
            std::vector<int> alive_indices(WH);
            for (int i = 0; i < WH; i++) {
                alive_indices[i] = i;
            }

            // 模型的一些定值
            int model_cascades = (std::max)(1 + static_cast<int>(std::ceil(std::log2(2 * model_scale))), 1);
            int model_grid_size = 128;
            float model_exp_step_factor = exp_step_factor;

            // 上传density_bitfield
            ncnn::VkMat density_bitfield_gpu;
            {
                ncnn::Mat density_bitfield_cpu(262144);
                int* data = (int*)density_bitfield_cpu.data;
                for (int i = 0; i < 262144; i++) {
                    data[i] = (int)model_density_bitfield[i];
                }
                cmd.record_clone(density_bitfield_cpu, density_bitfield_gpu, opt);
                //cmd.submit_and_wait();
                //cmd.reset();
            }
            

            ncnn::VkMat opacity_gpu(N_rays, (size_t)4u, 1, blob_vkallocator);
            ncnn::VkMat depth_gpu(N_rays, (size_t)4u, 1, blob_vkallocator);
            ncnn::VkMat rgb_gpu(3, N_rays, (size_t)4u, 1, blob_vkallocator);
            {
                {
                    std::vector<ncnn::VkMat> bindings(1);
                    bindings[0] = opacity_gpu;
                    std::vector<ncnn::vk_constant_type> constants(0);
                    ncnn::VkMat dispatcher;
                    dispatcher.w = opacity_gpu.w;
                    dispatcher.h = 1;
                    dispatcher.c = 1;
                    cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                }
                {
                    std::vector<ncnn::VkMat> bindings(1);
                    bindings[0] = depth_gpu;
                    std::vector<ncnn::vk_constant_type> constants(0);
                    ncnn::VkMat dispatcher;
                    dispatcher.w = depth_gpu.w;
                    dispatcher.h = 1;
                    dispatcher.c = 1;
                    cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                }
                {
                    std::vector<ncnn::VkMat> bindings(1);
                    bindings[0] = rgb_gpu;
                    std::vector<ncnn::vk_constant_type> constants(0);
                    ncnn::VkMat dispatcher;
                    dispatcher.w = rgb_gpu.w * rgb_gpu.h;
                    dispatcher.h = 1;
                    dispatcher.c = 1;
                    cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                }
                cmd.submit_and_wait();
                cmd.reset();
            }
            //ncnn::VkMat opacity_gpu, depth_gpu, rgb_gpu;
            //{
            //    ncnn::Mat opacity_cpu(N_rays, (size_t)4u, 1); opacity_cpu.fill<float>(0.0f);
            //    ncnn::Mat depth_cpu(N_rays, (size_t)4u, 1); depth_cpu.fill<float>(0.0f);
            //    ncnn::Mat rgb_cpu(3, N_rays, (size_t)4u, 1); rgb_cpu.fill<float>(0.0f);
            //    cmd.record_clone(opacity_cpu, opacity_gpu, opt);
            //    cmd.record_clone(depth_cpu, depth_gpu, opt);
            //    cmd.record_clone(rgb_cpu, rgb_gpu, opt);
            //    cmd.submit_and_wait();
            //    cmd.reset();
            //}



            ///////////////////////////////////////////////////////////////////////////////
            while (samples < max_samples) {

                int N_alive = alive_indices.size();
                if (N_alive == 0) {
                    break;
                }

                int N_samples = (std::max)((std::min)(N_rays / N_alive, 64), min_samples);
                samples += N_samples;


                /////////////////////////////////////////////////////////////////////////////// raymarching_test
                ncnn::Mat packed_info, ray_indices, deltas, ts;
                {
                    int model_max_samples = N_samples;

                    int model_N_rays = alive_indices.size();

                    // 上传alive_indices —— 后面会被修改，所以要动态更新上传
                    ncnn::VkMat alive_indices_gpu;
                    {
                        ncnn::Mat alive_indices_cpu(alive_indices.size());
                        int* data = (int*)alive_indices_cpu.data;
                        for (int i = 0; i < alive_indices.size(); i++) {
                            data[i] = alive_indices[i];
                        }
                        cmd.record_clone(alive_indices_cpu, alive_indices_gpu, opt);
                    }

                    // 开辟结果空间
                    ncnn::VkMat ray_indices_gpu(model_N_rays * model_max_samples, 4u, 1, blob_vkallocator);
                    ncnn::VkMat valid_mask_gpu(model_N_rays * model_max_samples, 4u, 1, blob_vkallocator);
                    ncnn::VkMat deltas_gpu(model_N_rays * model_max_samples, 4u, 1, blob_vkallocator);
                    ncnn::VkMat ts_gpu(model_N_rays * model_max_samples, 4u, 1, blob_vkallocator);
                    ncnn::VkMat samples_counter_gpu(model_N_rays, 4u, 1, blob_vkallocator);
                    ncnn::Mat ray_indices_cpu, valid_mask_cpu, deltas_cpu, ts_cpu, samples_counter_cpu;
                    {
                        {
                            std::vector<ncnn::VkMat> bindings(1);
                            bindings[0] = ray_indices_gpu;
                            std::vector<ncnn::vk_constant_type> constants(0);
                            ncnn::VkMat dispatcher;
                            dispatcher.w = ray_indices_gpu.w;
                            dispatcher.h = 1;
                            dispatcher.c = 1;
                            cmd.record_pipeline(clean_int_pipeline, bindings, constants, dispatcher);
                        }
                        {
                            std::vector<ncnn::VkMat> bindings(1);
                            bindings[0] = valid_mask_gpu;
                            std::vector<ncnn::vk_constant_type> constants(0);
                            ncnn::VkMat dispatcher;
                            dispatcher.w = valid_mask_gpu.w;
                            dispatcher.h = 1;
                            dispatcher.c = 1;
                            cmd.record_pipeline(clean_int_pipeline, bindings, constants, dispatcher);
                        }
                        {
                            std::vector<ncnn::VkMat> bindings(1);
                            bindings[0] = deltas_gpu;
                            std::vector<ncnn::vk_constant_type> constants(0);
                            ncnn::VkMat dispatcher;
                            dispatcher.w = deltas_gpu.w;
                            dispatcher.h = 1;
                            dispatcher.c = 1;
                            cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                        }
                        {
                            std::vector<ncnn::VkMat> bindings(1);
                            bindings[0] = ts_gpu;
                            std::vector<ncnn::vk_constant_type> constants(0);
                            ncnn::VkMat dispatcher;
                            dispatcher.w = ts_gpu.w;
                            dispatcher.h = 1;
                            dispatcher.c = 1;
                            cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                        }
                        {
                            std::vector<ncnn::VkMat> bindings(1);
                            bindings[0] = samples_counter_gpu;
                            std::vector<ncnn::vk_constant_type> constants(0);
                            ncnn::VkMat dispatcher;
                            dispatcher.w = samples_counter_gpu.w;
                            dispatcher.h = 1;
                            dispatcher.c = 1;
                            cmd.record_pipeline(clean_int_pipeline, bindings, constants, dispatcher);
                        }
                        cmd.submit_and_wait();
                        cmd.reset();
                    }
                    //ncnn::Mat ray_indices_cpu(model_N_rays * model_max_samples); ray_indices_cpu.fill<int>(0);
                    //ncnn::Mat valid_mask_cpu(model_N_rays * model_max_samples); valid_mask_cpu.fill<int>(0);
                    //ncnn::Mat deltas_cpu(model_N_rays * model_max_samples); deltas_cpu.fill<float>(0.0f);
                    //ncnn::Mat ts_cpu(model_N_rays * model_max_samples); ts_cpu.fill<float>(0.0f);
                    //ncnn::Mat samples_counter_cpu(model_N_rays); samples_counter_cpu.fill<int>(0);
                    //ncnn::VkMat ray_indices_gpu, valid_mask_gpu, deltas_gpu, ts_gpu, samples_counter_gpu;
                    //cmd.record_clone(ray_indices_cpu, ray_indices_gpu, opt);
                    //cmd.record_clone(valid_mask_cpu, valid_mask_gpu, opt);
                    //cmd.record_clone(deltas_cpu, deltas_gpu, opt);
                    //cmd.record_clone(ts_cpu, ts_gpu, opt);
                    //cmd.record_clone(samples_counter_cpu, samples_counter_gpu, opt);
                    //cmd.submit_and_wait();
                    //cmd.reset();

                    
                    // 跑kernel
                    {
                        std::vector<ncnn::VkMat> bindings(10);
                        bindings[0] = rays_o_gpu; // float
                        bindings[1] = rays_d_gpu; // float
                        bindings[2] = hits_t_gpu; // float
                        bindings[3] = alive_indices_gpu; // int
                        bindings[4] = density_bitfield_gpu; // int

                        bindings[5] = ray_indices_gpu; // int
                        bindings[6] = valid_mask_gpu; // int
                        bindings[7] = deltas_gpu; // float
                        bindings[8] = ts_gpu; // float
                        bindings[9] = samples_counter_gpu; // int

                        std::vector<ncnn::vk_constant_type> constants(5);
                        constants[0].i = model_cascades;
                        constants[1].i = model_grid_size;
                        constants[2].f = model_scale;
                        constants[3].f = model_exp_step_factor;
                        constants[4].i = model_max_samples;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = alive_indices.size();
                        dispatcher.h = 1;
                        dispatcher.c = 1;

                        cmd.record_pipeline(raymarching_test_pipeline, bindings, constants, dispatcher);

                        cmd.record_clone(ray_indices_gpu, ray_indices_cpu, opt);
                        cmd.record_clone(valid_mask_gpu, valid_mask_cpu, opt);
                        cmd.record_clone(deltas_gpu, deltas_cpu, opt);
                        cmd.record_clone(ts_gpu, ts_cpu, opt);
                        cmd.record_clone(samples_counter_gpu, samples_counter_cpu, opt);
                        cmd.submit_and_wait();
                        cmd.reset();
                    }


                    /////////////////////////////////////////////////////////////////////////////// 后处理
                    {
                        packed_info.create(2, model_N_rays, (size_t)4u, 1);
                        {
                            int* p1 = (int*)samples_counter_cpu.data;
                            int* p2 = (int*)packed_info.data;
                            int cumsum = 0;
                            for (int i = 0; i < model_N_rays; i++) {
                                cumsum += p1[i];
                                p2[2 * i + 0] = cumsum - p1[i];
                                p2[2 * i + 1] = p1[i];
                            }
                        }

                        int num = 0;
                        {
                            int* p = (int*)valid_mask_cpu.data;
                            for (int i = 0; i < model_N_rays * model_max_samples; i++) {
                                if (p[i] > 0) {
                                    num += 1;
                                }
                            }
                        }

                        ray_indices.create(num);
                        deltas.create(num);
                        ts.create(num);

                        {
                            int* p = (int*)valid_mask_cpu.data;

                            int* s1 = (int*)ray_indices_cpu.data;
                            float* s2 = (float*)deltas_cpu.data;
                            float* s3 = (float*)ts_cpu.data;

                            int* d1 = (int*)ray_indices.data;
                            float* d2 = (float*)deltas.data;
                            float* d3 = (float*)ts.data;

                            for (int i = 0; i < model_N_rays * model_max_samples; i++) {
                                if (p[i] > 0) {
                                    *d1++ = (int)s1[i];
                                    *d2++ = (float)s2[i];
                                    *d3++ = (float)s3[i];
                                }
                            }
                        }
                    }
                }


                if (ray_indices.w == 0) {
                    break;
                }

                ncnn::Mat ray_o_local = rays_o;
                ncnn::Mat ray_d_local(3, ray_indices.w);

                ncnn::Mat rays_d_cpu;
                cmd.record_clone(rays_d_gpu, rays_d_cpu, opt);
                cmd.submit_and_wait();
                cmd.reset();

                ncnn::Mat xyzs(3, ray_indices.w);
                {
                    int* ray_indices_p = (int*)ray_indices.data;
                    float* rays_d_p = (float*)rays_d_cpu.data;
                    float* ray_d_local_p = (float*)ray_d_local.data;
                    float* ray_o_local_p = (float*)ray_o_local.data;
                    float* ts_p = (float*)ts.data;
                    float* xyzs_p = (float*)xyzs.data;

                    for (int i = 0; i < ray_indices.w; i++) {
                        int pos = ray_indices_p[i];
                        ray_d_local_p[3 * i + 0] = rays_d_p[3 * pos + 0];
                        ray_d_local_p[3 * i + 1] = rays_d_p[3 * pos + 1];
                        ray_d_local_p[3 * i + 2] = rays_d_p[3 * pos + 2];

                        xyzs_p[3 * i + 0] = ray_o_local_p[0] + ts_p[i] * ray_d_local_p[3 * i + 0];
                        xyzs_p[3 * i + 1] = ray_o_local_p[1] + ts_p[i] * ray_d_local_p[3 * i + 1];
                        xyzs_p[3 * i + 2] = ray_o_local_p[2] + ts_p[i] * ray_d_local_p[3 * i + 2];
                    }
                }


                ncnn::Mat dirs = ray_d_local;


                /////////////////////////////////////////////////////////////////////////////// density
                ncnn::Mat h_cpu;
                ncnn::Mat sigmas_cpu;
                {
                    float model_xyz_min = -1.0 * model_scale;
                    float model_xyz_max = 1.0 * model_scale;
                    ncnn::Mat x = xyzs;
                    {
                        float* p = (float*)x.data;
                        for (int i = 0; i < x.h * x.w; i++) {
                            p[i] = (p[i] - model_xyz_min) / (model_xyz_max - model_xyz_min);
                        }
                    }


                    /////////////////////////////////////////////////////////////////////////////// pos_encoder
                    
                    ncnn::VkMat output_embedding_gpu;
                    {
                        // input_pos
                        ncnn::VkMat input_pos_gpu;
                        cmd.record_clone(x, input_pos_gpu, opt);

                        // params
                        // 已经提前上传好了

                        // output_embedding
                        output_embedding_gpu.create(hash_encoder_out_dim, x.h, (size_t)4u, 1, blob_vkallocator);
                        {
                            std::vector<ncnn::VkMat> bindings(1);
                            bindings[0] = output_embedding_gpu;
                            std::vector<ncnn::vk_constant_type> constants(0);
                            ncnn::VkMat dispatcher;
                            dispatcher.w = output_embedding_gpu.w * output_embedding_gpu.h;
                            dispatcher.h = 1;
                            dispatcher.c = 1;
                            cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                        }
                        //{
                        //    ncnn::Mat output_embedding_cpu(hash_encoder_out_dim, x.h); output_embedding_cpu.fill<float>(0.0f);
                        //    cmd.record_clone(output_embedding_cpu, output_embedding_gpu, opt);
                        //}

                        // hash_map_sizes
                        ncnn::VkMat hash_map_sizes_gpu;
                        {
                            ncnn::Mat hash_map_sizes(16);
                            int* p = (int*)hash_map_sizes.data;
                            for (int i = 0; i < 16; i++) {
                                p[i] = hash_encoder_hash_map_sizes[i];
                            }
                            cmd.record_clone(hash_map_sizes, hash_map_sizes_gpu, opt);
                        }

                        // offsets
                        ncnn::VkMat offsets_gpu;
                        {
                            ncnn::Mat offsets(16);
                            int* p = (int*)offsets.data;
                            for (int i = 0; i < 16; i++) {
                                p[i] = hash_encoder_offsets[i];
                            }
                            cmd.record_clone(offsets, offsets_gpu, opt);
                        }

                        // 跑kernel
                        {
                            std::vector<ncnn::VkMat> bindings(5);
                            bindings[0] = input_pos_gpu; // float
                            bindings[1] = params_gpu; // float
                            bindings[2] = output_embedding_gpu; // float
                            bindings[3] = hash_map_sizes_gpu; // int
                            bindings[4] = offsets_gpu; // int

                            std::vector<ncnn::vk_constant_type> constants(2);
                            constants[0].i = x.h;
                            constants[1].i = hash_encoder_out_dim;

                            ncnn::VkMat dispatcher;
                            dispatcher.w = x.h;
                            dispatcher.h = hash_encoder_hash_level;
                            dispatcher.c = 1;

                            cmd.record_pipeline(hash_encoder_pipeline, bindings, constants, dispatcher);

                            cmd.submit_and_wait();
                            cmd.reset();
                        }
                    }
                    
                    


                    /////////////////////////////////////////////////////////////////////////////// xyz_encoder
                    {
                        
                        ncnn::Mat output_embedding_cpu;
                        cmd.record_clone(output_embedding_gpu, output_embedding_cpu, opt);
                        cmd.submit_and_wait();
                        cmd.reset();

                        //double t1 = ncnn::get_current_time();
                        { // 超级无敌慢，急需优化
                            //ncnn::Mat test;
                            ncnn::Extractor ex = xyz_encoder_net.create_extractor();
                            ex.input("in0", output_embedding_cpu);
                            //ex.extract("3", test, 1);
                            ex.extract("out0", h_cpu, 1);
                            //{
                            //    double sum = 0.0;
                            //    float* data = (float*)test.data;
                            //    for (int i = 0; i < test.h * test.w; i++) {
                            //        sum += double(data[i]);
                            //    }
                            //    // printf("(%d,%d,%d),%.10lf\n", test.c, test.h, test.w, sum);
                            //}
                        }
                        //double t2 = ncnn::get_current_time();
                        //printf("cost:%f\n",t2-t1);

                        // printf("%d\n", output_embedding_cpu.h);

                        /*
                        // 跑kernel
                        double t1 = ncnn::get_current_time();
                        {
                            ncnn::VkMat inter_out(64, output_embedding_gpu.h, (size_t)4u, 1, blob_vkallocator);
                            {
                                std::vector<ncnn::VkMat> bindings(3);
                                bindings[0] = output_embedding_gpu;
                                bindings[1] = xyz_encoder_weight_0_gpu;
                                bindings[2] = inter_out;

                                std::vector<ncnn::vk_constant_type> constants(0);

                                ncnn::VkMat dispatcher;
                                dispatcher.w = 64;
                                dispatcher.h = output_embedding_gpu.h;
                                dispatcher.c = 1;

                                cmd.record_pipeline(xyz_encoder_weight_0_gemm_pipeline, bindings, constants, dispatcher);

                                cmd.submit_and_wait();
                                cmd.reset();
                            }
                            ncnn::VkMat final_out(16, output_embedding_gpu.h, (size_t)4u, 1, blob_vkallocator);
                            {
                                std::vector<ncnn::VkMat> bindings(3);
                                bindings[0] = inter_out;
                                bindings[1] = xyz_encoder_weight_1_gpu;
                                bindings[2] = final_out;

                                std::vector<ncnn::vk_constant_type> constants(0);

                                ncnn::VkMat dispatcher;
                                dispatcher.w = 16;
                                dispatcher.h = output_embedding_gpu.h;
                                dispatcher.c = 1;

                                cmd.record_pipeline(xyz_encoder_weight_1_gemm_pipeline, bindings, constants, dispatcher);

                                cmd.submit_and_wait();
                                cmd.reset();
                            }
                            cmd.record_clone(final_out, h_cpu, opt);
                            cmd.submit_and_wait();
                            cmd.reset();
                        }
                        double t2 = ncnn::get_current_time();
                        printf("cost:%f\n", t2 - t1);
                        */
                    }

                    sigmas_cpu.create(h_cpu.h);
                    {
                        float* src = (float*)h_cpu.data;
                        float* dst = (float*)sigmas_cpu.data;
                        for (int i = 0; i < h_cpu.h; i++) {
                            *dst = exp(*src);
                            src += 16;
                            dst += 1;
                        }
                    }
                }

                ncnn::Mat d = dirs;
                {
                    float* p = (float*)d.data;
                    for (int i = 0; i < d.h; i++) {
                        float norm = sqrt(pow(p[0], 2) + pow(p[1], 2) + pow(p[2], 2));
                        p[0] /= norm;
                        p[1] /= norm;
                        p[2] /= norm;
                        p += 3;
                    }
                }


                /////////////////////////////////////////////////////////////////////////////// dir_encoder
                ncnn::VkMat dir_encoder_output_embedding_gpu;
                {
                    // 准备输入输出
                    ncnn::VkMat input_dir_gpu;
                    cmd.record_clone(d, input_dir_gpu, opt);

                    dir_encoder_output_embedding_gpu.create(16, d.h, (size_t)4u, 1, blob_vkallocator);
                    {
                        std::vector<ncnn::VkMat> bindings(1);
                        bindings[0] = dir_encoder_output_embedding_gpu;
                        std::vector<ncnn::vk_constant_type> constants(0);
                        ncnn::VkMat dispatcher;
                        dispatcher.w = dir_encoder_output_embedding_gpu.w * dir_encoder_output_embedding_gpu.h;
                        dispatcher.h = 1;
                        dispatcher.c = 1;
                        cmd.record_pipeline(clean_float_pipeline, bindings, constants, dispatcher);
                    }
                    //{
                    //    ncnn::Mat dir_encoder_output_embedding_cpu(16, d.h); dir_encoder_output_embedding_cpu.fill<float>(0.0f);
                    //    cmd.record_clone(dir_encoder_output_embedding_cpu, dir_encoder_output_embedding_gpu, opt);
                    //}

                    // 跑kernel
                    {
                        std::vector<ncnn::VkMat> bindings(2);
                        bindings[0] = input_dir_gpu; // float
                        bindings[1] = dir_encoder_output_embedding_gpu; // float

                        std::vector<ncnn::vk_constant_type> constants(0);

                        ncnn::VkMat dispatcher;
                        dispatcher.w = d.h;
                        dispatcher.h = 1;
                        dispatcher.c = 1;

                        cmd.record_pipeline(dir_encoder_pipeline, bindings, constants, dispatcher);

                        cmd.submit_and_wait();
                        cmd.reset();
                    }
                }

                ncnn::Mat d_cpu;
                cmd.record_clone(dir_encoder_output_embedding_gpu, d_cpu, opt);
                cmd.submit_and_wait();
                cmd.reset();


                /////////////////////////////////////////////////////////////////////////////// rgb_net
                ncnn::Mat rgbs_cpu;
                {
                    { // 超级无敌慢，急需优化
                        ncnn::Extractor ex = rgb_net.create_extractor();
                        ex.input("in0", d_cpu);
                        ex.input("in1", h_cpu);
                        ex.extract("out0", rgbs_cpu);
                    }
                }


                /////////////////////////////////////////////////////////////////////////////// composite_test
                {
                    // 准备输入输出
                    ncnn::VkMat sigmas_gpu;
                    cmd.record_clone(sigmas_cpu, sigmas_gpu, opt);

                    ncnn::VkMat rgbs_gpu;
                    cmd.record_clone(rgbs_cpu, rgbs_gpu, opt);

                    ncnn::VkMat deltas_gpu;
                    cmd.record_clone(deltas, deltas_gpu, opt);

                    ncnn::VkMat ts_gpu;
                    cmd.record_clone(ts, ts_gpu, opt);

                    ncnn::VkMat pack_info_gpu;
                    cmd.record_clone(packed_info, pack_info_gpu, opt);

                    ncnn::VkMat alive_indices_gpu;
                    {
                        ncnn::Mat alive_indices_cpu(alive_indices.size());
                        int* data = (int*)alive_indices_cpu.data;
                        for (int i = 0; i < alive_indices.size(); i++) {
                            data[i] = alive_indices[i];
                        }
                        cmd.record_clone(alive_indices_cpu, alive_indices_gpu, opt);
                    }

                    // 跑kernel
                    {
                        std::vector<ncnn::VkMat> bindings(9);
                        bindings[0] = sigmas_gpu; // float
                        bindings[1] = rgbs_gpu; // float
                        bindings[2] = deltas_gpu; // float
                        bindings[3] = ts_gpu; // float
                        bindings[4] = pack_info_gpu; // int
                        bindings[5] = alive_indices_gpu; // int
                        bindings[6] = opacity_gpu; // float
                        bindings[7] = depth_gpu; // float
                        bindings[8] = rgb_gpu; // float

                        std::vector<ncnn::vk_constant_type> constants(1);
                        constants[0].f = T_threshold;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = alive_indices.size();
                        dispatcher.h = 1;
                        dispatcher.c = 1;

                        cmd.record_pipeline(composite_test_pipeline, bindings, constants, dispatcher);

                        ncnn::Mat alive_indices_cpu;
                        cmd.record_clone(alive_indices_gpu, alive_indices_cpu, opt);
                        cmd.submit_and_wait();
                        cmd.reset();

                        {
                            int* data = (int*)alive_indices_cpu.data;
                            int sum = 0;
                            for (int i = 0; i < alive_indices_cpu.w; i++) {
                                if (data[i] >= 0) {
                                    sum += 1;
                                }
                            }

                            //int m1 = 0;
                            //for (int i = 0; i < alive_indices_cpu.w; i++) {
                            //    if (data[i] == -1) {
                            //        m1 += 1;
                            //    }
                            //}
                            //int m2 = 0;
                            //for (int i = 0; i < alive_indices_cpu.w; i++) {
                            //    if (data[i] == -2) {
                            //        m2 += 1;
                            //    }
                            //}
                            //printf("%d,%d,%d\n", sum, m1, m2);

                            alive_indices.clear();
                            alive_indices.resize(sum);

                            int j = 0;
                            for (int i = 0; i < alive_indices_cpu.w; i++) {
                                if (data[i] >= 0) {
                                    alive_indices[j++] = data[i];
                                }
                            }
                        }
                    }
                }
            }


            /////////////////////////////////////////////////////////////////////////////// 处理背景
            float rgb_bg = 0.0f;
            if (exp_step_factor == 0)
                rgb_bg = 1.0f;

            ncnn::Mat rgb_cpu, opacity_cpu;
            cmd.record_clone(rgb_gpu, rgb_cpu, opt);
            cmd.record_clone(opacity_gpu, opacity_cpu, opt);
            cmd.submit_and_wait();
            cmd.reset();

            {
                float* rgb_p = (float*)rgb_cpu.data;
                float* opa_p = (float*)opacity_cpu.data;
                for (int i = 0; i < WH; i++) {
                    rgb_p[3 * i + 0] += rgb_bg * (1.0 - opa_p[i]);
                    rgb_p[3 * i + 1] += rgb_bg * (1.0 - opa_p[i]);
                    rgb_p[3 * i + 2] += rgb_bg * (1.0 - opa_p[i]);
                }
            }


            /////////////////////////////////////////////////////////////////////////////// Log
            double end_time = ncnn::get_current_time();
            printf("generating: %3d/%d \t\tfps:%.2f\n", pose_idx + 1, POSE, 1000 / (end_time - start_time));


            /////////////////////////////////////////////////////////////////////////////// 保存图像
            std::vector<unsigned char> result(img_w * img_h * 3);
            float* data = (float*)rgb_cpu.data;
            for (int i = 0; i < img_w * img_h * 3; ++i) {
                result[i] = (unsigned char)(data[i] * 255.0);
            }
            stbi_write_png(("results/" + std::to_string(pose_idx + 1) + ".png").c_str(), img_w, img_h, 3, result.data(), img_w * 3);

        }



        /////////////////////////////////////////////////////////////////////////////// 释放
        {
            rgb_net.clear();
            xyz_encoder_net.clear();

            //delete xyz_encoder_weight_0_gemm_pipeline;
            //delete xyz_encoder_weight_1_gemm_pipeline;
            delete composite_test_pipeline;
            delete dir_encoder_pipeline;
            delete hash_encoder_pipeline;
            delete raymarching_test_pipeline;
            delete ray_aabb_intersect_pipeline;
            delete clean_int_pipeline;
            delete clean_float_pipeline;

            gemm->destroy_pipeline(opt);
            delete gemm;

            vkdev->reclaim_blob_allocator(blob_vkallocator);
            vkdev->reclaim_staging_allocator(blob_vkallocator);

            // pc上n卡有bug，所以不destory了
            // ncnn::destroy_gpu_instance();
        }
    }

    return 0;
}