#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__global__ void render(float4 *imageBuffer, int max_x, int max_y, int splits)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    int start_x = tile_x * (max_x / splits);
    int start_y = tile_y * (max_y / splits);

    int tileSpan_x = max_x / splits;
    int tileSpan_y = max_y / splits;

    if (tile_x == (gridDim.x - 1))
        tileSpan_x = (max_x - tile_x * tileSpan_x);

    if (tile_y == (gridDim.y - 1))
        tileSpan_y = (max_y - tile_y * tileSpan_y);

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    float4 color;
    color.x = tile_x * 0.025f;
    color.y = tile_y * 0.025f;
    color.z = 0.7f;
    color.w = 1.0f;

    if (start_x >= max_x || start_y >= max_y)
        return;

    for (int x = start_x; x < start_x + tileSpan_x && x < max_x; x++)
    {
        for (int y = start_y; y < start_y + tileSpan_y && y < max_y; y++)
        {
            imageBuffer[x * max_y + y] = color;
        }
    }
}
