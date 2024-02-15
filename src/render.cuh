#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__global__ void render(float4 *imageBuffer, int max_x, int max_y)
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

    // if(tile_x == 8 && tile_y == 8){
    //     if(thread_x == 31 && thread_y == 31){
    //         printf("%d * (%d / %d) + %d\n", thread_y, tileSpan_y, blockDim.y, tileStart_y);
    //     }
    // }

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
            imageBuffer[x * max_y + y] = color;
        }
    }

    // if(tile_x == 15 && tile_y == 15){
    //     if(thread_x == 31 && thread_y == 31){
    //         printf("Tile start: [%d, %d]\nTile span: [%d, %d]\nThread start: [%d, %d]\nThread span: [%d, %d]\n", tileStart_x, tileStart_y, tileSpan_x, tileSpan_y, threadStart_x, threadStart_y, threadSpan_x, threadSpan_y);
    //         printf("block: [%d, %d, %d], grid: [%d, %d, %d]\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    //     }
    // }
}
