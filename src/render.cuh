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

    int tileStart_x = tile_x * (max_x / gridDim.x);
    int tileStart_y = tile_y * (max_y / gridDim.x);

    int tileSpan_x = max_x / gridDim.x;
    int tileSpan_y = max_y / gridDim.x;
    
    if (tile_x == (gridDim.x - 1))
        tileSpan_x = (max_x - tile_x * tileSpan_x);

    if (tile_y == (gridDim.y - 1))
        tileSpan_y = (max_y - tile_y * tileSpan_y);

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int threadStart_x = thread_x * (tileSpan_x / blockDim.x) + tileStart_x;
    int threadStart_y = thread_y * (tileSpan_y / blockDim.y) + tileStart_y;

    int threadSpan_x = tileSpan_x / blockDim.x;
    int threadSpan_y = tileSpan_y / blockDim.y;

    if (thread_x == (blockDim.x - 1))
        threadSpan_x = (tileSpan_x - thread_x * threadSpan_x);

    if (thread_y == (blockDim.y - 1))
        threadSpan_y = (tileSpan_y - thread_y * threadSpan_y);

    float4 color;
    color.x = (tile_x * 0.025f + tile_y * 0.025f) / 2.0f;
    color.y = thread_y * 0.025f;
    color.z = thread_x * 0.025f;
    color.w = 1.0f;

    if (threadStart_x >= max_x || threadStart_y >= max_y)
        return;

    for (int x = threadStart_x; x < threadStart_x + threadSpan_x && x < max_x; x++)
    {
        for (int y = threadStart_y; y < threadStart_y + threadSpan_y && y < max_y; y++)
        {
            glm::vec2 ssc = glm::vec2((((float)x) / max_x) * 2.0f - 1.0f, (((float)y) / max_y) * 2.0f - 1.0f);
            /* Per-Pixel operations */
            imageBuffer[x * max_y + y] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for(int splat = 0; splat < 1000; splat++){
                glm::vec4 position = glm::vec4(sd[splat].fields.position[0], sd[splat].fields.position[1], sd[splat].fields.position[2], 1.0f);
                position = perspective * position;
                if(position[0] < -1 || position[0] > 1 || position[1] < -1 || position[1] > 1 || position[2] > 0){
                    continue;
                }
                if(glm::distance(ssc, glm::vec2(position[0], position[1])) < 0.02){
                    imageBuffer[x * max_y + y].x += clip(0.5 + SH_C0 * sd[splat].fields.SH[0], 0.0f, 1.0f);
                    imageBuffer[x * max_y + y].y += clip(0.5 + SH_C0 * sd[splat].fields.SH[1], 0.0f, 1.0f);
                    imageBuffer[x * max_y + y].z += clip(0.5 + SH_C0 * sd[splat].fields.SH[2], 0.0f, 1.0f);
                    imageBuffer[x * max_y + y].w += clip(sd[splat].fields.opacity, 0.0f, 1.0f) / 10;
                }
            }
            // imageBuffer[x * max_y + y] = color;
            
        }
    }

    // if(tile_x == 15 && tile_y == 15){
    //     if(thread_x == 31 && thread_y == 31){
    //         printf("Tile start: [%d, %d]\nTile span: [%d, %d]\nThread start: [%d, %d]\nThread span: [%d, %d]\n", tileStart_x, tileStart_y, tileSpan_x, tileSpan_y, threadStart_x, threadStart_y, threadSpan_x, threadSpan_y);
    //         printf("block: [%d, %d, %d], grid: [%d, %d, %d]\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    //     }
    // }
}
