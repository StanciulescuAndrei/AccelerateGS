#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "PLYReader.h"


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
	float fovx,
    float4 * conic_opacity, 
    float3 * rgb, 
    float2 * image_point,
    int * radius, // Radius in pixels
    float * depth,
    int * num_tiles_overlap,
    const int SCREEN_WIDTH,
    const int SCREEN_HEIGHT,
    dim3 grid,
	int renderMode,
	bool * renderMask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_splats)
        return;

	radius[idx] = 0;
    num_tiles_overlap[idx] = 0;

	if(renderMask[idx] == false)
		return;

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

    /* Compute world-space covariance !!! ALREADY DONE IN DATA LOADING */

    /* Compute 2D screen-space covariance matrix */
	float tan_fovy = tanf(fovy * 0.5f);
	float tan_fovx = tanf(fovx * 0.5f);

	float focal_x = SCREEN_WIDTH / (2.0f * tan_fovx);
	float focal_y = SCREEN_HEIGHT / (2.0f * tan_fovy);

	float3 cov = computeCov2D(pOrig, focal_x, focal_y, tan_fovx, tan_fovy, sd[idx].fields.covariance, view);

	if((renderMode & 0b1111) == 1){
		cov.x = 0.91f;
		cov.y = 0.0f;
		cov.z = 0.91f;
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

	// Store some useful helper data for the next steps.
	depth[idx] = position_viewport.z;
	radius[idx] = my_radius;
	image_point[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, sd[idx].fields.opacity };
	num_tiles_overlap[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	if(renderMode>>4 == 0){
		glm::vec3 result = computeColorFromSH(idx, 3, sd[idx].fields, camPos);
		rgb[idx] = {result.x, result.y, result.z};
	}
	else if(renderMode>>4 == 1){
		rgb[idx] = {position_viewport.z / 20.0f, position_viewport.z / 20.0f, position_viewport.z / 20.0f};
	}
	else if(renderMode>>4 == 2){
		glm::vec3 norm_cov = glm::normalize(glm::vec3(cov.x, cov.y, cov.z));
		rgb[idx] = {norm_cov.x, norm_cov.y, norm_cov.z};
	}

	// if(cov.x < 0.3 || cov.z < 0.3){
	// 	rgb[idx] = {1.0f, 0.0f, 0.0f};
	// }


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

	/* Each thread in the block has its own rank */
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
		
		/* Each thread collects one set of data to shared memory */
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
