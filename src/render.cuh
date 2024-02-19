#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "PLYReader.h"

__device__ const float SH_C0 = 0.28209479177387814;

__device__ float clip(float in, float min_val, float max_val){
    return min(max_val, max(in, min_val));
}

__global__ void render(SplatData * sd, float4 *imageBuffer, int max_x, int max_y, glm::mat4 perspective, int num_splats)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    int thread_x = tile_x * blockDim.x + threadIdx.x;
    int thread_y = tile_y * blockDim.y + threadIdx.y;

    /* Debug here */
    if(thread_x == 0 && thread_y){
        // printf("%d\n", sizeof(float));
    }

    if(thread_x >= max_x || thread_y >= max_y)
        return;

    glm::vec2 ssc = glm::vec2((((float)thread_x) / max_x) * 2.0f - 1.0f, (((float)thread_y) / max_y) * 2.0f - 1.0f);
    /* Per-Pixel operations */
    imageBuffer[thread_x * max_y + thread_y] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for(int splat = 0; splat < 100; splat++){
        glm::vec4 position = glm::vec4(sd[splat].fields.position[0], sd[splat].fields.position[1], sd[splat].fields.position[2], 1.0f);
        position = perspective * position;
        if(position[0] < -1 || position[0] > 1 || position[1] < -1 || position[1] > 1 || position[2] > 0){
            continue;
        }
        if(glm::distance(ssc, glm::vec2(position[0], position[1])) < 0.02){
            imageBuffer[thread_x * max_y + thread_y].x += clip(0.5 + SH_C0 * sd[splat].fields.SH[0], 0.0f, 1.0f);
            imageBuffer[thread_x * max_y + thread_y].y += clip(0.5 + SH_C0 * sd[splat].fields.SH[1], 0.0f, 1.0f);
            imageBuffer[thread_x * max_y + thread_y].z += clip(0.5 + SH_C0 * sd[splat].fields.SH[2], 0.0f, 1.0f);
            imageBuffer[thread_x * max_y + thread_y].w += clip(sd[splat].fields.opacity, 0.0f, 1.0f);
        }
    }

    // if(tile_x == 15 && tile_y == 15){
    //     if(thread_x == 31 && thread_y == 31){
    //         printf("Tile start: [%d, %d]\nTile span: [%d, %d]\nThread start: [%d, %d]\nThread span: [%d, %d]\n", tileStart_x, tileStart_y, tileSpan_x, tileSpan_y, threadStart_x, threadStart_y, threadSpan_x, threadSpan_y);
    //         printf("block: [%d, %d, %d], grid: [%d, %d, %d]\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    //     }
    // }
}
