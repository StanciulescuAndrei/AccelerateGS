#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "PLYReader.h"

#define BLOCK_X 16
#define BLOCK_Y 16

__device__ const float SH_C0 = 0.28209479177387814;

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
__host__ void printMat(glm::mat3 m){
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
	printf("-----------------------------\n");
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

__device__ float3 computeCov2D(const glm::vec4& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const glm::mat4 viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	glm::vec4 t = viewmatrix * mean;

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		1.0f / t.z, 0.0f, -(1.0f * t.x) / (t.z * t.z),
		0.0f, 1.0f / t.z, -(1.0f * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0],
		viewmatrix[0][1], viewmatrix[1][1], viewmatrix[2][1],
		viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const float * scale, float mod, const float * rot, float* cov3D)
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

__global__ void cumulativeSum(int num_splats, int * num_tiles_overlap){
	for(int i = 1; i < num_splats; i++){
		num_tiles_overlap[i] = num_tiles_overlap[i] + num_tiles_overlap[i-1];
	}
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
	getRect(image_point[idx], radius[idx], rect_min, rect_max, grid);

	int offset = 0;
	if(idx > 0) offset += cumulative_sum_overlaps[idx - 1];

	for(int x = rect_min.x; x < rect_max.x; x++){
		for(int y = rect_min.y; y < rect_max.y; y++){
			uint32_t tile_id = y * grid.x + x;
			sort_keys[offset] = (uint64_t)0;
			sort_keys[offset] += tile_id;
			sort_keys[offset] <<= 32;
			sort_keys[offset] += (*((uint32_t *)&(depth[idx]))); // Move the bits of the depth, which is float, to the lowest 32 bits of the sorting key

			gaussian_ids[offset] = idx;

			offset++;
		}
	}
}

__global__ void preprocessGaussians(int num_splats, SplatData * sd, 
    glm::mat4 projection, 
    glm::mat4 modelview, 
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_splats)
        return;

    radius[idx] = 0;
    num_tiles_overlap[idx] = 0;

    /* Check if splat is in frustum */
    glm::vec4 pOrig = glm::vec4(sd[idx].fields.position[0], sd[idx].fields.position[1], sd[idx].fields.position[2], 1.0f);
    glm::vec4 position = projection * modelview * pOrig;
    if(position[0] < -1 || position[0] > 1 || position[1] < -1 || position[1] > 1 || position[2] > 0){
        return;
    }

    /* Compute world-space covariance */
    float cov3D[6];
    computeCov3D(sd[idx].fields.scale, 1.0f, sd[idx].fields.rotation, cov3D);

    /* Compute 2D screen-space covariance matrix */
	float3 cov = computeCov2D(pOrig, 1.0f, 1.0f, 1.0f, 1.0f, cov3D, modelview);

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
	float2 point_image = { (position.x + 1.0f) * 0.5f * SCREEN_WIDTH, (position.y + 1.0f) * 0.5f * SCREEN_HEIGHT };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
	
    // glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
    rgb[idx] = {
        clip(0.5 + SH_C0 * sd[idx].fields.SH[0], 0.0f, 1.0f),
        clip(0.5 + SH_C0 * sd[idx].fields.SH[1], 0.0f, 1.0f),
        clip(0.5 + SH_C0 * sd[idx].fields.SH[2], 0.0f, 1.0f)
    };

	// Store some useful helper data for the next steps.
	depth[idx] = position.z;
	radius[idx] = my_radius;
	image_point[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, sd[idx].fields.opacity };
	num_tiles_overlap[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

}

__global__ void debugInfo(int num_splats, SplatData * sd, 
    glm::mat4 projection, 
    glm::mat4 modelview, 
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
    printf("Conics  : %f, %f, %f, %f\n", conic_opacity[0].x, conic_opacity[0].y, conic_opacity[0].z, conic_opacity[0].w);
    printf("RGB     : %f, %f, %f\n", rgb[0].x, rgb[0].y, rgb[0].z);
    printf("I. P.   : %f, %f\n", image_point[0].x, image_point[0].y);
    printf("Radius  : %d\n", radius[0]);
    printf("Depth   : %f\n", depth[0]);
    printf("Overlap : %d\n", num_tiles_overlap[0]);
    printf("----------------------------------------------\n");
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

	float T = 1.0f;
	float pixColor[4] = { 0, 0, 0, 0 };

    if(thread_x >= SCREEN_WIDTH || thread_y >= SCREEN_HEIGHT)
        return;

    /* Per-Pixel operations */
    imageBuffer[(SCREEN_HEIGHT - thread_y - 1) * SCREEN_WIDTH + thread_x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for(int i = min_range; i < max_range; i++){
		uint32_t splat_id = splat_ids[i];

        float2 position = image_point[splat_id];
		float2 d = { position.x - thread_x, position.y - thread_y };

		float4 con_o = conic_opacity[splat_id];
		float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

		if (power > 0.0f) continue;

		float alpha = min(0.99f, con_o.w * exp(power));

		float test_T = T * (1 - alpha);
		if (test_T < 0.0001f)
		{
			break;
		}

		// Eq. (3) from 3D Gaussian splatting paper.
		pixColor[0] += rgb[splat_id].x * alpha * T;
		pixColor[1] += rgb[splat_id].y * alpha * T;
		pixColor[2] += rgb[splat_id].z * alpha * T;

		T = test_T;

    }
	imageBuffer[(SCREEN_HEIGHT - thread_y - 1) * SCREEN_WIDTH + thread_x].x = pixColor[0];
	imageBuffer[(SCREEN_HEIGHT - thread_y - 1) * SCREEN_WIDTH + thread_x].y = pixColor[1];
	imageBuffer[(SCREEN_HEIGHT - thread_y - 1) * SCREEN_WIDTH + thread_x].z = pixColor[2];
	imageBuffer[(SCREEN_HEIGHT - thread_y - 1) * SCREEN_WIDTH + thread_x].w = 1.0f;
}
