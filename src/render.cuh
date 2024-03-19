#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Dense>

#include "PLYReader.h"
#include "GaussianOctree.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

#define LINE_BLOCK 256

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ float clip(float in, float min_val, float max_val){
    return min(max_val, max(in, min_val));
}

__host__ void printMat(glm::mat4 m){
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
	printf("-----------------------------\n");
}
__device__ void printMat(glm::mat3 m){
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
	printf("-----------------------------\n");
}

__device__ glm::vec3 computeColorFromSH(int idx, int deg, const SplatData::Fields & sdFields, glm::vec3 & campos)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = glm::make_vec3(sdFields.position);
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 * sh = (glm::vec3*) sdFields.SH;

	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	return glm::max(result, 0.0f);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__device__ float3 computeCov2D(const glm::vec4& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const glm::mat4 view, float debug = false)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	glm::vec4 t = view * mean;

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view[0][0], view[1][0], view[2][0],
		view[0][1], view[1][1], view[2][1],
		view[0][2], view[1][2], view[2][2]);

	// W = glm::transpose(W);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	// cov[0][0] += 0.3f;
	// cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const float * scale, float mod, const float * rot, float* cov3D, glm::vec3 & normal)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale[0];
	S[1][1] = mod * scale[1];
	S[2][2] = mod * scale[2];

    glm::vec4 q = glm::vec4(rot[0], rot[1], rot[2], rot[3]);
    q = q * (1.0f / glm::length(q));
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// Compute the normal orientation
	if(scale[0] < scale[1] && scale[0] < scale[2]){
		normal = glm::vec3(1.0f, 0.0f, 0.0f);
	}
	else if(scale[1] < scale[0] && scale[1] < scale[2]){
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
	}
	else{
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
	}

	normal = R * normal;

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__global__ void duplicateGaussians(int num_splats, 
    float2 * image_point,
    int * radius, // Radius in pixels
    float * depth,
	int * cumulative_sum_overlaps,
	uint64_t * sort_keys,
	uint32_t * gaussian_ids,
    dim3 grid)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_splats) return;

	uint2 rect_min, rect_max;
	if(radius[idx] <= 0) return;

	getRect(image_point[idx], radius[idx], rect_min, rect_max, grid);

	int offset = 0;
	if(idx > 0) offset += cumulative_sum_overlaps[idx - 1];

	for(int x = rect_min.x; x < rect_max.x; x++){
		for(int y = rect_min.y; y < rect_max.y; y++){
			uint64_t splat_key = y * grid.x + x;
			splat_key <<= 32;
			splat_key |= (*((uint32_t *)&(depth[idx])));
			sort_keys[offset] = splat_key; // Move the bits of the depth, which is float, to the lowest 32 bits of the sorting key

			gaussian_ids[offset] = idx;

			offset++;
		}
	}
}

__global__ void preprocessGaussians(int num_splats, SplatData * sd, 
    glm::mat4 projection, 
    glm::mat4 view, 
	glm::vec3 camPos,
	float fovy,
    float4 * conic_opacity, 
    float3 * rgb, 
    float2 * image_point,
    int * radius, // Radius in pixels
    float * depth,
    int * num_tiles_overlap,
    const int SCREEN_WIDTH,
    const int SCREEN_HEIGHT,
    dim3 grid,
	int renderMode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_splats)
        return;

    radius[idx] = 0;
    num_tiles_overlap[idx] = 0;

    /* Check if splat is in frustum */
    glm::vec4 pOrig = glm::vec4(sd[idx].fields.position[0], sd[idx].fields.position[1], sd[idx].fields.position[2], 1.0f);
	glm::vec4 p_hom = projection * pOrig;
	float p_w = 1.0f / (p_hom.w + 0.000000001f);
	glm::vec3 p_proj = p_hom * p_w;
    glm::vec4 position_viewport = view * pOrig;
	// float p_w = 1.0f / (position.w + 0.000000001f);
	// position = position * p_w;

	position_viewport *= (-1.0f);

    if(position_viewport.z <= 0.2f){
        return;
    }

	if(fabsf(p_proj.x) > 1.3f || fabsf(p_proj.y) > 1.3f){
		return;
	}

    /* Compute world-space covariance */
    float cov3D[6];
	glm::vec3 splatNormal;

    computeCov3D(sd[idx].fields.scale, 1.0f, sd[idx].fields.rotation, cov3D, splatNormal);

    /* Compute 2D screen-space covariance matrix */
	float tan_fovy = tanf(fovy * 0.5f);
	float tan_fovx = tan_fovy * 16.0f / 9.0f;

	float focal_x = SCREEN_WIDTH / (2.0f * tan_fovx);
	float focal_y = SCREEN_HEIGHT / (2.0f * tan_fovy);

	float3 cov = computeCov2D(pOrig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, view);

	if((renderMode & 0b1111) == 1){
		cov.x = 0.31f;
		cov.y = 0.0f;
		cov.z = 0.31f;
	}

    // Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { (-p_proj.x + 1.0f) * 0.5f * SCREEN_WIDTH, (-p_proj.y + 1.0f) * 0.5f * SCREEN_HEIGHT };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	if(renderMode>>4 == 0){
		glm::vec3 result = computeColorFromSH(idx, 3, sd[idx].fields, camPos);
		rgb[idx] = {result.x, result.y, result.z};
	}
	else if(renderMode>>4 == 1){
		rgb[idx] = {1.0f - position_viewport.z / 10.0f, 1.0f - position_viewport.z / 10.0f, 1.0f - position_viewport.z / 10.0f};
	}
	else if(renderMode>>4 == 2){
		glm::vec3 norm_cov = glm::normalize(glm::vec3(cov.x, cov.y, cov.z));
		rgb[idx] = {splatNormal.x, splatNormal.y, splatNormal.z};
	}

	// if(cov.x < 0.3 || cov.z < 0.3){
	// 	rgb[idx] = {1.0f, 0.0f, 0.0f};
	// }

	


	// Store some useful helper data for the next steps.
	depth[idx] = position_viewport.z;
	radius[idx] = my_radius;
	image_point[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, sd[idx].fields.opacity };
	num_tiles_overlap[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

}

__global__ void debugInfo(int num_splats, SplatData * sd, 
    glm::mat4 projection, 
    glm::mat4 view, 
    float4 * conic_opacity, 
    float3 * rgb, 
    float2 * image_point,
    int * radius, // Radius in pixels
    float * depth,
    int * num_tiles_overlap,
    const int SCREEN_WIDTH,
    const int SCREEN_HEIGHT,
    dim3 grid)
{
	int num_proc = 0;
	for(int i=0;i<num_splats;i++){
		if(radius[i] > 0)
			num_proc++;
	}
	float perc = (float) num_proc * 100.0f / num_splats;
	printf("Processing %d out of %d gaussians (%f%%)\n-------------------------------------------\n", num_proc, num_splats, perc);
    // printf("Conics  : %f, %f, %f, %f\n", conic_opacity[0].x, conic_opacity[0].y, conic_opacity[0].z, conic_opacity[0].w);
    // printf("RGB     : %f, %f, %f\n", rgb[0].x, rgb[0].y, rgb[0].z);
    // printf("I. P.   : %f, %f\n", image_point[0].x, image_point[0].y);
    // printf("Radius  : %d\n", radius[0]);
    // printf("Depth   : %f\n", depth[0]);
    // printf("Overlap : %d\n", num_tiles_overlap[0]);
    // printf("----------------------------------------------\n");
}

__global__ void getTileRanges(uint64_t * sorted_keys, int array_len, uint32_t * tile_range_min, uint32_t * tile_range_max){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= array_len) return;

	uint32_t tile_id = sorted_keys[idx] >> 32;

	if(idx == 0){
		tile_range_min[tile_id] = 0;
	}
	else{
		uint32_t prev_tile_id = sorted_keys[idx - 1] >> 32;
		if(tile_id != prev_tile_id){
			tile_range_max[prev_tile_id] = idx;
			tile_range_min[tile_id] = idx;
		}
	}

	if(idx == array_len - 1){
		tile_range_max[tile_id] = array_len;
	}
}

__global__ void render(int num_splats, SplatData * sd, 
    float4 * conic_opacity, 
    float3 * rgb, 
    float2 * image_point,
	float * depth,
	uint32_t * tile_range_min,
	uint32_t * tile_range_max,
	uint32_t * splat_ids,
    const int SCREEN_WIDTH,
    const int SCREEN_HEIGHT,
    dim3 grid, float4 * imageBuffer)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

	uint32_t tile_id = blockIdx.y * gridDim.x + blockIdx.x;

	uint32_t min_range = tile_range_min[tile_id];
	uint32_t max_range = tile_range_max[tile_id];

    int thread_x = tile_x * blockDim.x + threadIdx.x;
    int thread_y = tile_y * blockDim.y + threadIdx.y;

	int thread_rank = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ float2 positions[BLOCK_SIZE];
	__shared__ float4 conics[BLOCK_SIZE];
	__shared__ uint32_t splat_data_id[BLOCK_SIZE];
	// __shared__ float3 colors[BLOCK_SIZE];

	float T = 1.0f;
	float pixColor[4] = { 0, 0, 0, 0 };

	bool validPixel = !(thread_x >= SCREEN_WIDTH || thread_y >= SCREEN_HEIGHT);

	/* Per-Pixel operations */
	int array_offset = min_range;
	while(array_offset < max_range){
		
		if(array_offset + thread_rank < max_range){
			splat_data_id[thread_rank] = splat_ids[array_offset + thread_rank];
			positions[thread_rank] = image_point[splat_data_id[thread_rank]];
			conics[thread_rank] = conic_opacity[splat_data_id[thread_rank]];
		}

		__syncthreads();

		if(validPixel && T > 0.0001f){
			for(int i = 0; i < min(BLOCK_SIZE, max_range - array_offset); i++){
				float2 d = { positions[i].x - thread_x, positions[i].y - thread_y };
				float4 con_o = conics[i];
				float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

				if (power > 0.0f) continue;

				float alpha = fminf(0.99f, con_o.w * expf(power));

				if(alpha < 1.0f / 255.0f) continue;

				float test_T = T * (1 - alpha);
				if (test_T < 0.0001f)
				{
					break;
				}

				// Eq. (3) from 3D Gaussian splatting paper.
				float scaling = alpha * T;
				float3 collected_rgb = rgb[splat_data_id[i]];

				pixColor[0] = fmaf(collected_rgb.x, scaling, pixColor[0]);
				pixColor[1] = fmaf(collected_rgb.y, scaling, pixColor[1]);
				pixColor[2] = fmaf(collected_rgb.z, scaling, pixColor[2]);
				// pixColor[0] = fmaf(1.0f - depth[splat_data_id[i]] / 20.0f, scaling, pixColor[0]);
				// pixColor[1] = fmaf(1.0f - depth[splat_data_id[i]] / 20.0f, scaling, pixColor[1]);
				// pixColor[2] = fmaf(1.0f - depth[splat_data_id[i]] / 20.0f, scaling, pixColor[2]);

				// pixColor[0] = depth[splat_data_id[i]];
				// pixColor[1] = depth[splat_data_id[i]];
				// pixColor[2] = depth[splat_data_id[i]];

				T = test_T;
			}
		}

		__syncthreads();
		array_offset += BLOCK_SIZE;

	}
	if(validPixel){
		imageBuffer[thread_y * SCREEN_WIDTH + thread_x].x = pixColor[0];
		imageBuffer[thread_y * SCREEN_WIDTH + thread_x].y = pixColor[1];
		imageBuffer[thread_y * SCREEN_WIDTH + thread_x].z = pixColor[2];
		imageBuffer[thread_y * SCREEN_WIDTH + thread_x].w = 1.0f;
	}
}
