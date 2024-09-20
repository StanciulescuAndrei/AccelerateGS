#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <memory>

#include <iomanip>
#include <ctime>
#include <sstream>

#include <iostream>
#include <pthread.h>

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <chrono>

#include "OctreeSerializer.cuh"
#include "render.cuh"

#include "cuda_common/helper_cuda.h"

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "CameraLoader.h"

#include <deque>

#include <thrust/count.h>
#include <thrust/device_vector.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "raster_helper.cuh"
#include "HybridVH.h"


#define N 100000000
#define MAX_ERR 1e-6

int SCREEN_WIDTH = 1920;
int SCREEN_HEIGHT = 1080;

const int FPS_COUNTER_REFRESH = 60;

glm::mat3 cameraRotation;
glm::mat4 modelview;
glm::mat4 perspective;

glm::vec3 cameraPosition = glm::vec3(0.0f);
float angleDirection = 0.0f;
glm::vec3 lookDirection = glm::vec3(1.0f, 0.0f, 0.0f);
glm::vec3 rightDirection = glm::vec3(0.0f, 0.0f, 1.0f);
const float movement_step = 0.1f;

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

int markForRender(bool *renderMask, std::vector<HybridVH *> & nodes, int renderLevel, glm::vec3 &cameraPosition, float fovy, int SW, float dpt, int numSplats)
{
	int rendered = 0;
    omp_set_num_threads(6);
    // #pragma omp parallel for
	for(int i = 0; i < nodes.size(); i++){
        HybridVH * tree_root = nodes[i];
        std::deque<HybridVH *> process_queue;
        process_queue.push_back(tree_root);
        while(!process_queue.empty()){
            HybridVH * node = process_queue.front();
            process_queue.pop_front();
            if (renderLevel == -1)
            {
                int shouldRenderNode = 0;
                if (node == nullptr)
                    continue;
                /* Easiest implementation, maximum projection by distance */
                float S = glm::length(node->coverage[0] - node->coverage[1]);
                float D = glm::length((node->coverage[0] + node->coverage[1]) / 2.0f - cameraPosition);

                float P = S / D * (SW / fovy);

                shouldRenderNode = (P > dpt);

                if (shouldRenderNode)
                { // is node big enough on the screen?
                    if (node->containedSplats->size() > 0)
                    {
                        for (uint32_t splat : *(node->containedSplats))
                            renderMask[splat] = true;
                    }
                    else
                    {
                        for(auto child : node ->children){
                            process_queue.push_back(child);
                        }
                    }
                }
                else
                {
                    if (node->representative != 0)
                    {
                        renderMask[node->representative] = true;
                    }
                    else
                    {
                        if(node->children.size() == 0){
                            for(auto splat : *(node->containedSplats)){
                                renderMask[splat] = 1;
                            }
                        }
                        for(auto child : node->children){
                            process_queue.push_back(child);
                        }
                    }
                }
            }
            else
            {
                if (node->level == renderLevel && node->representative != 0)
                {
                    renderMask[node->representative] = true;
                }
                if (node->level < renderLevel && node->isLeaf)
                {
                    for (uint32_t splat : *(node->containedSplats)){
                        renderMask[splat] = true;
                    }
                }
                else if(node->level < renderLevel){
                    for(HybridVH * child : node->children){
                        process_queue.push_back(child);
                    }
                }
            }
        }
	}
    
    for(int i = 0; i < numSplats; i++){
        if(renderMask[i]){
            rendered++;
        }
    }

    return rendered;
}
 
template <typename T>
int markForRender(bool *renderMask, T * node, int renderLevel, glm::vec3 &cameraPosition, float fovy, int SW, float dpt, int numSplats)
{
	int rendered = 0;
        T * tree_root = node;
        std::deque<T *> process_queue;
        process_queue.push_back(tree_root);
        while(!process_queue.empty()){
            T * node = process_queue.front();
            process_queue.pop_front();
            if (renderLevel == -1)
            {
                int shouldRenderNode = 0;
                if (node == nullptr)
                    continue;
                /* Easiest implementation, maximum projection by distance */
                float S = glm::length(node->coverage[0] - node->coverage[1]);
                float D = glm::length((node->coverage[0] + node->coverage[1]) / 2.0f - cameraPosition);

                float P = S / D * (SW / fovy);

                shouldRenderNode = (P > dpt);

                if (shouldRenderNode)
                { // is node big enough on the screen?
                    if (node->containedSplats->size() > 0)
                    {
                        for (uint32_t splat : *(node->containedSplats))
                            renderMask[splat] = true;
                    }
                    else
                    {
                        for(auto child : node ->children){
                            process_queue.push_back(child);
                        }
                    }
                }
                else
                {
                    if (node->representative != 0)
                    {
                        renderMask[node->representative] = true;
                    }
                    else
                    {
                        if(node->children.size() == 0){
                            for(auto splat : *(node->containedSplats)){
                                renderMask[splat] = 1;
                            }
                        }
                        for(auto child : node->children){
                            process_queue.push_back(child);
                        }
                    }
                }
            }
            else
            {
                if (node->level == renderLevel && node->representative != 0)
                {
                    renderMask[node->representative] = true;
                }
                if (node->level < renderLevel && node->isLeaf)
                {
                    for (uint32_t splat : *(node->containedSplats)){
                        renderMask[splat] = true;
                    }
                }
                else if(node->level < renderLevel){
                    for(T * child : node->children){
                        process_queue.push_back(child);
                    }
                }
            }
        }
    
    for(int i = 0; i < numSplats; i++){
        if(renderMask[i]){
            rendered++;
        }
    }

    return rendered;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if(key == GLFW_KEY_Z && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += cameraRotation* glm::vec3(0.0f, movement_step, 0.0f);
    }
    if(key == GLFW_KEY_X && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += cameraRotation* glm::vec3(0.0f, -movement_step, 0.0f);
    }
    if(key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += movement_step * cameraRotation* rightDirection;
    }
    if(key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition -= movement_step * cameraRotation* rightDirection;
    }
    if(key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += movement_step * cameraRotation* lookDirection;
    }
    if(key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition -= movement_step * cameraRotation* lookDirection;
    }
    // if(key == GLFW_KEY_Q && (action == GLFW_REPEAT || action == GLFW_PRESS)){
    //     angleDirection -= 0.01f;
    //     lookDirection  = glm::vec3(sin(angleDirection), 0.f, cos(angleDirection));
    //     rightDirection = glm::vec3(-cos(angleDirection), 0.f, sin(angleDirection));
    // }
    // if(key == GLFW_KEY_E && (action == GLFW_REPEAT || action == GLFW_PRESS)){
    //     angleDirection += 0.01f;
    //     lookDirection  = glm::vec3(sin(angleDirection), 0.f, cos(angleDirection));
    //     rightDirection = glm::vec3(-cos(angleDirection), 0.f, sin(angleDirection));
    // }
    if(key == GLFW_KEY_U && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraIndex = std::max(0, cameraIndex - 1);
    }
    if(key == GLFW_KEY_I && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraIndex = std::min(300, cameraIndex + 1);
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void initGLContextAndWindow(GLFWwindow** window){
    
    
    glfwSetErrorCallback(error_callback);
 
    if (!glfwInit())
        exit(EXIT_FAILURE);
 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
 
    *window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "AccelerateGS", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
 
    glfwSetKeyCallback(*window, key_callback);
    glfwSetFramebufferSizeCallback(*window, framebuffer_size_callback);
 
    glfwMakeContextCurrent(*window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(0);

    setupIMGui(window);
}

struct ThreadPayload{
    SpacePartitioningBase * spacePartitioningRoot;
    std::vector<SplatData> * sd;
    int * num_elements;
    volatile int * progress;
};

void * spacePartitioningThread(void * input){
    ThreadPayload * payload = static_cast<ThreadPayload *>(input);
    payload->spacePartitioningRoot->buildVHStructure(*(payload->sd), *(payload->num_elements), payload->progress);
    pthread_exit(NULL);
}

template <typename T>
void splitTree(std::vector<T*> & nodes, SpacePartitioningBase * spbase){
    /*
        Split structure in shallower subtrees.
        This allows independent marking on each subtree on the GPU.
    */
    T * sparse_node = new T();
    sparse_node->isLeaf = true;
    sparse_node->representative = 0;
    sparse_node->levelType = OctreeLevel;

    std::deque<T *> q_nodes;
    q_nodes.push_back((T*)(spbase));

    while(!q_nodes.empty()){
        T * crt_node = q_nodes.front();
        q_nodes.pop_front();
        for(T* child : crt_node->children){
            if(child->levelType == BipartitionLevel){
                nodes.push_back(child);
            }
        }
        for(T* child : crt_node->children){
            if(child->levelType == OctreeLevel){
                if(child->isLeaf){
                    for(auto splat : *(child->containedSplats))
                    sparse_node->containedSplats->push_back(splat);
                }
                else{
                    q_nodes.push_back(child);
                }
            }    
        }
    }
    nodes.push_back(sparse_node);
    printf("Sparse node size: %d\n", sparse_node->containedSplats->size());

}

int main(){
    GLFWwindow* window;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    loadCameraFile("../../models/train/cameras.json");
    loadGenericProperties(SCREEN_WIDTH, SCREEN_HEIGHT, fovx, fovy);

    loadApplicationConfig("../config.cfg", renderConfig);

    numCameraPositions = cameraData.size();

    initGLContextAndWindow(&window);

    /* Load splat scene data from file */
    std::vector<SplatData> sd;
    bool * renderMask;
    int num_elements = 0;
    int res = loadSplatData("../../models/train/point_cloud/iteration_small/point_cloud.ply", sd, &num_elements);
    printf("Loaded %d splats from file\n", num_elements);

    const uint32_t orig_num_splats = num_elements;

    const uint32_t maxDuplicatedGaussians = num_elements * 64;

    // First of all, build da octree
    begin = std::chrono::steady_clock::now();
    #if defined(_OPENMP)
        printf("Using OpenMP, yey\n");
    #endif

    /* OpenGL configuration */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 16);      // 4-byte pixel alignment

    glClearColor(0, 0, 0, 0);                   // background color
    glClearStencil(0);                          // clear stencil buffer
    glClearDepth(1.0f);                         // 0 is near, 1 is far
    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  


    volatile int progress = 0;
    float progressmax = 16.0f;
    SpacePartitioningBase * spacePartitioningRoot = (nullptr);
    if(renderConfig.structure == std::string("octree")){
        spacePartitioningRoot = new GaussianOctree();
        progressmax = 256.0f;
    }
    else if(renderConfig.structure == std::string("bvh")){
        spacePartitioningRoot = new GaussianBVH();
    }
    else if(renderConfig.structure == std::string("hybrid")){
        spacePartitioningRoot = new HybridVH();
        progressmax = 256.0f;
    }

    if(spacePartitioningRoot == nullptr){
        printf("This ain't good lol....\n");
    }

    octreeLevel = renderConfig.octreeLevel;


    /* Compute space partitioning in a separate PThread */
    pthread_t t_id;
    ThreadPayload payload;
    payload.num_elements = &num_elements;
    payload.progress = &progress;
    payload.spacePartitioningRoot = spacePartitioningRoot;
    payload.sd = &sd;

#if true
    pthread_create(&t_id, NULL, spacePartitioningThread, (void *)(&payload));

    while(progress!=1024){
        /* Clear color and depth buffers */
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        buildLoadingInterface(progress / progressmax);
        renderInterface();
        /* Swap buffers and handle GLFW events */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    pthread_join(t_id, NULL);
#else
    spacePartitioningRoot->buildVHStructure(sd, num_elements, &progress);
#endif
    // std::vector<HybridVH *> nodes;
    // splitTree<HybridVH>(nodes, spacePartitioningRoot);

    // /* Put node pointers into array for CUDA processing */

    // size_t totalStorageSize = 0;
    // size_t nodeHeaderSize = sizeof(CUDATreeNode);

    // for(int i = 0; i < nodes.size(); i++){
    //     HybridVH * tree_root = nodes[i];
    //     std::deque<HybridVH *> process_queue;
    //     process_queue.push_back(tree_root);
    //     while(!process_queue.empty()){
    //         HybridVH * node = process_queue.front();

    //         totalStorageSize += 1;
            
    //         process_queue.pop_front();
    //         for(HybridVH * child : node->children){
    //             process_queue.push_back(child);
    //         }

    //     }
	// }

    // /* Get the necessary memory to serialize in RAM the array of subtrees */
    // CUDATreeNode * storageBlock = (CUDATreeNode *)malloc(totalStorageSize * sizeof(CUDATreeNode));
    // CUDATreeNode * cudaStorageBlock = NULL;
    // uint32_t * cuda_roots = NULL;

    // size_t currentMemoryPosition = 0;
    // std::vector<uint32_t> roots;

    // for(int i = 0; i < nodes.size(); i++){
    //     HybridVH * tree_root = nodes[i];
    //     roots.push_back(currentMemoryPosition);
    //     std::deque<HybridVH *> process_queue;
    //     process_queue.push_back(tree_root);
    //     while(!process_queue.empty()){
    //         HybridVH * node = process_queue.back();
    //         process_queue.pop_back();

    //         size_t child1Pos = 0, child2Pos = 0;
    //         if(node->children.size() > 0){
    //             child1Pos = process_queue.size() + currentMemoryPosition + 1;
    //         }
    //         if(node->children.size() > 1){
    //             child2Pos = process_queue.size() + currentMemoryPosition + 2;
    //         }

    //         storageBlock[currentMemoryPosition].childrenIndices[0] = child1Pos;
    //         storageBlock[currentMemoryPosition].childrenIndices[1] = child2Pos;

    //         for(int s = 0; s < sizeof(storageBlock[currentMemoryPosition].splatIds) / sizeof(uint32_t); s++){
    //             storageBlock[currentMemoryPosition].splatIds[s] = 0;
    //         }

    //         for(int s = 0; s < std::min(node->containedSplats->size(), sizeof(storageBlock[currentMemoryPosition].splatIds) / sizeof(uint32_t)); s++){
    //             if(node->containedSplats->size() > sizeof(storageBlock[currentMemoryPosition].splatIds) / sizeof(uint32_t)){
    //                 std::cout<<node->containedSplats->size()<<" "<<sizeof(storageBlock[currentMemoryPosition].splatIds) / sizeof(uint32_t)<<std::endl;
    //             }
    //             storageBlock[currentMemoryPosition].splatIds[s] = (*(node->containedSplats))[s];
    //         }

    //         storageBlock[currentMemoryPosition].representative = node->representative;
    //         storageBlock[currentMemoryPosition].flags = node->isLeaf;
    //         storageBlock[currentMemoryPosition].level = node->level;

    //         glm::vec3 center = (node->coverage[0] + node->coverage[1]) / 2.0f;

    //         storageBlock[currentMemoryPosition].center.x = center.x;
    //         storageBlock[currentMemoryPosition].center.y = center.y;
    //         storageBlock[currentMemoryPosition].center.z = center.z; 

    //         storageBlock[currentMemoryPosition].diagonal = glm::length((node->coverage[0] - node->coverage[1]));

    //         currentMemoryPosition++;

    //         for(HybridVH * child : node->children){
    //             process_queue.push_back(child);
    //         }

    //     }
	// }

    // printf("Built %d subtrees\n", roots.size());

    printf("Done building space partitioning\n");

    num_elements = sd.size();
    renderMask = (bool *)malloc(sizeof(bool) * num_elements);
    memset(renderMask, 0, sizeof(bool) * num_elements);

    end = std::chrono::steady_clock::now();
    int octreeTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    printf("Octree built in %f s\n", octreeTime / 1000.0f);
    
    printf("Number of splats: %d\n", num_elements);

    // checkCudaErrors(cudaMalloc(&cudaStorageBlock, sizeof(CUDATreeNode) * totalStorageSize));
    // assert(cudaStorageBlock != NULL);
    // checkCudaErrors(cudaMemcpy((void *)cudaStorageBlock, (void *)storageBlock, sizeof(CUDATreeNode) * totalStorageSize, cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMalloc(&cuda_roots, sizeof(uint32_t) * roots.size()));
    // assert(cuda_roots != NULL);
    // checkCudaErrors(cudaMemcpy((void *)cuda_roots, (void *)roots.data(), sizeof(uint32_t) * roots.size(), cudaMemcpyHostToDevice));

    /* Allocate and send splat data to GPU memory */
    SplatData * d_sd;
    checkCudaErrors(cudaMalloc(&d_sd, sizeof(SplatData) * num_elements));
    assert(d_sd != NULL);
    checkCudaErrors(cudaMemcpy((void*)d_sd, (void*) sd.data(), sizeof(SplatData) * num_elements, cudaMemcpyHostToDevice));

    /* Allocate additional data buffers */
    float4 * d_conic_opacity;
    float3 * d_rgb;
    float2 * d_image_point;
    int * d_radius;
    float * d_depth;
    int * d_overlap;
    int * d_overlap_sums;
    bool * d_renderMask;
    float * d_cov3ds;

    checkCudaErrors(cudaMalloc(&d_conic_opacity, sizeof(float4) * num_elements));
    assert(d_conic_opacity != NULL);

    checkCudaErrors(cudaMalloc(&d_rgb, sizeof(float3) * num_elements));
    assert(d_rgb != NULL);

    checkCudaErrors(cudaMalloc(&d_image_point, sizeof(float2) * num_elements));
    assert(d_image_point != NULL);

    checkCudaErrors(cudaMalloc(&d_radius, sizeof(int) * num_elements));
    assert(d_radius != NULL);

    checkCudaErrors(cudaMalloc(&d_depth, sizeof(float) * num_elements));
    assert(d_depth != NULL);

    checkCudaErrors(cudaMalloc(&d_overlap, sizeof(int) * num_elements));
    assert(d_overlap != NULL);

    checkCudaErrors(cudaMalloc(&d_overlap_sums, sizeof(int) * num_elements));
    assert(d_overlap_sums != NULL);

    checkCudaErrors(cudaMalloc(&d_cov3ds, sizeof(int) * num_elements * 6));
    assert(d_cov3ds != NULL);

    checkCudaErrors(cudaMalloc(&d_renderMask, sizeof(bool) * num_elements));
    assert(d_renderMask != NULL);
    checkCudaErrors(cudaMemcpy(d_renderMask, renderMask, sizeof(bool) * num_elements, cudaMemcpyHostToDevice));

    thrust::device_ptr<bool> d_renderMask_devPtr(d_renderMask);

    dim3 block(BLOCK_X, BLOCK_Y, 1); // One thread per pixel!
    dim3 grid(SCREEN_WIDTH / BLOCK_X + 1, SCREEN_HEIGHT / BLOCK_Y + 1, 1);

    uint32_t * d_tile_range_min;
    uint32_t * d_tile_range_max;

    checkCudaErrors(cudaMalloc(&d_tile_range_min, sizeof(uint32_t) * grid.x * grid.y));
    checkCudaErrors(cudaMalloc(&d_tile_range_max, sizeof(uint32_t) * grid.x * grid.y));

    uint64_t * d_sort_keys_in;
    uint64_t * d_sort_keys_out;
    uint32_t * d_sort_ids_in;
    uint32_t * d_sort_ids_out;
    checkCudaErrors(cudaMalloc(&d_sort_keys_in, sizeof(uint64_t) * maxDuplicatedGaussians));
    checkCudaErrors(cudaMalloc(&d_sort_keys_out, sizeof(uint64_t) * maxDuplicatedGaussians));
    checkCudaErrors(cudaMalloc(&d_sort_ids_in, sizeof(uint32_t) * maxDuplicatedGaussians));
    checkCudaErrors(cudaMalloc(&d_sort_ids_out, sizeof(uint32_t) * maxDuplicatedGaussians));

    /* Set up resources for texture writing */
    GLuint pboId;
    GLuint texId;
    GLfloat * imageData = new GLfloat[SCREEN_HEIGHT * SCREEN_WIDTH * 4];

    struct cudaGraphicsResource * cuda_pbo_resource;
    void * d_pbo_buffer = NULL;

    // Initialize the texture
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, (GLvoid*)imageData);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Initialize PBO
    glGenBuffers(1, &pboId);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, SCREEN_HEIGHT * SCREEN_WIDTH * 4 * sizeof(float), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Prepare CUDA interop
    checkCudaErrors(cudaMalloc(&d_pbo_buffer, 4 * SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(float)));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pboId, cudaGraphicsRegisterFlagsNone));

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 200000000;

    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    

    /* Very basic FPS metrics */
    int currentFPSIndex = 0;

    getCameraParameters(0, cameraPosition, cameraRotation);

    begin = std::chrono::steady_clock::now();

    cudaEvent_t kernelStart;
    cudaEvent_t kernelEnd;

    checkCudaErrors(cudaEventCreate(&kernelStart));
    checkCudaErrors(cudaEventCreate(&kernelEnd));

    /* Main program loop */
    while (!glfwWindowShouldClose(window))
    {
        /* Clear color and depth buffers */
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);


        /* Bind the texture and Pixel Buffer */
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
        glBindTexture(GL_TEXTURE_2D, texId);
        
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_FLOAT, 0);

        /* Map the OpenGL resources to a CUDA memory location */
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        float4* dataPointer = nullptr;
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dataPointer, &num_bytes, cuda_pbo_resource));
        assert(num_bytes >= SCREEN_HEIGHT * SCREEN_WIDTH * 4 * sizeof(float));
        assert(dataPointer != nullptr);

        auto softwareRasterizer = [&] (int forcedCameraIndex){
            /* --------- RENDERING ------------*/
            if(cameraMode == 1){
                getCameraParameters(forcedCameraIndex, cameraPosition, cameraRotation);
            }

            modelview = glm::lookAt(cameraPosition, cameraRotation * glm::vec3(0.0f, 0.0f, 1.0f) + cameraPosition, cameraRotation * glm::vec3(0.0f, 1.0f, 0.0f));
            perspective = glm::perspective(fovy, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.009f, 100.0f) * modelview;

            Frustum f;
            computeFrustum(f, perspective);

            int renderMode = (selectedViewMode<<4) + renderPrimitive;

            checkCudaErrors(cudaMemset(d_renderMask, 0, sizeof(bool) * num_elements));
            // checkCudaErrors(cudaMemset(d_renderMask, 1, sizeof(bool) * orig_num_splats / 4));
            memset(renderMask, 0, sizeof(bool) * num_elements);

            if(renderConfig.structure == std::string("octree")) 
                markForRender<GaussianOctree>(renderMask, static_cast<GaussianOctree*>(spacePartitioningRoot), autoLevel ? -1 : renderLevel, cameraPosition, fovy, SCREEN_WIDTH, diagonalProjectionThreshold, num_elements);
            else if(renderConfig.structure == std::string("bvh"))
                markForRender<GaussianBVH>(renderMask, static_cast<GaussianBVH*>(spacePartitioningRoot), autoLevel ? -1 : renderLevel, cameraPosition, fovy, SCREEN_WIDTH, diagonalProjectionThreshold, num_elements);
            else
                markForRender<HybridVH>(renderMask, static_cast<HybridVH*>(spacePartitioningRoot), autoLevel ? -1 : renderLevel, cameraPosition, fovy, SCREEN_WIDTH, diagonalProjectionThreshold, num_elements);

            // markForRender(renderMask, nodes, autoLevel ? -1 : renderLevel, cameraPosition, fovy, SCREEN_WIDTH, diagonalProjectionThreshold, num_elements);
            cudaMemcpy(d_renderMask, renderMask, sizeof(bool) * num_elements, cudaMemcpyHostToDevice);

            checkCudaErrors(cudaEventRecord(kernelStart));
            // CUDAmarkForRender<<<roots.size() / 256 + 1, 256>>>(d_renderMask, cudaStorageBlock, cuda_roots, roots.size(), cameraPosition, fovy, SCREEN_WIDTH, diagonalProjectionThreshold, f, useFrustumCulling, (!autoLevel)*renderLevel);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaEventRecord(kernelEnd));
            checkCudaErrors(cudaEventSynchronize(kernelEnd));
            checkCudaErrors(cudaEventElapsedTime(&traversalTime, kernelStart, kernelEnd));
            avgTraversal.insert(traversalTime);

            checkCudaErrors(cudaEventRecord(kernelStart));

            preprocessGaussians<<<num_elements / LINE_BLOCK + 1, LINE_BLOCK>>>(num_elements, d_sd, perspective, modelview, cameraPosition, fovy, fovx, d_conic_opacity, d_rgb, d_image_point, d_radius, d_depth, d_overlap, SCREEN_WIDTH, SCREEN_HEIGHT, grid, renderMode, d_renderMask);
            checkCudaErrors(cudaDeviceSynchronize());

            renderedSplats = thrust::count(d_renderMask_devPtr, d_renderMask_devPtr + num_elements, true);
            
            // Run inclusive prefix sum
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_overlap, d_overlap_sums, num_elements);

            int totalDuplicateGaussians = 0;
            checkCudaErrors(cudaMemcpy(&totalDuplicateGaussians, d_overlap_sums + num_elements - 1, sizeof(int), cudaMemcpyDeviceToHost));

            totalDuplicateGaussians = min(totalDuplicateGaussians, maxDuplicatedGaussians);

            /* Populate sorting keys array */
            duplicateGaussians<<<num_elements / LINE_BLOCK + 1, LINE_BLOCK>>>(num_elements, d_image_point, d_radius, d_depth, d_overlap_sums, d_sort_keys_in, d_sort_ids_in, grid);
            checkCudaErrors(cudaDeviceSynchronize());

            /* TODO: determine highest MSB to pass to sorting, so we don't use all 64 bits */
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sort_keys_in, d_sort_keys_out, d_sort_ids_in, d_sort_ids_out, totalDuplicateGaussians);

            checkCudaErrors(cudaMemset(d_tile_range_min, 0, sizeof(uint32_t) * grid.x * grid.y));
            checkCudaErrors(cudaMemset(d_tile_range_max, 0, sizeof(uint32_t) * grid.x * grid.y));

            getTileRanges<<<(totalDuplicateGaussians) / LINE_BLOCK + 1, LINE_BLOCK>>>(d_sort_keys_out, totalDuplicateGaussians, d_tile_range_min, d_tile_range_max);
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaEventRecord(kernelEnd));
            checkCudaErrors(cudaEventSynchronize(kernelEnd));
            checkCudaErrors(cudaEventElapsedTime(&prepTime, kernelStart, kernelEnd));
            avgPrep.insert(prepTime);

            checkCudaErrors(cudaEventRecord(kernelStart));
            render<<<grid, block>>>(num_elements, d_sd, d_conic_opacity, d_rgb, d_image_point, d_depth, d_tile_range_min, d_tile_range_max, d_sort_ids_out, SCREEN_WIDTH, SCREEN_HEIGHT, grid, dataPointer);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaEventRecord(kernelEnd));
            checkCudaErrors(cudaEventSynchronize(kernelEnd));
            checkCudaErrors(cudaEventElapsedTime(&renderTime, kernelStart, kernelEnd));
            avgRender.insert(renderTime);
        };

        auto saveRenderRoutine = [&](const char * filename){
            std::vector<float> floatPixelData(SCREEN_HEIGHT * SCREEN_WIDTH * 4);
            std::vector<unsigned char> pixelData(SCREEN_HEIGHT * SCREEN_WIDTH * 4);

            cudaMemcpy(floatPixelData.data(), dataPointer, SCREEN_HEIGHT * SCREEN_WIDTH * 4 * sizeof(float), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < pixelData.size(); ++i) {
                pixelData[i] = static_cast<unsigned char>(std::min(1.0f, floatPixelData[i]) * 255.0f);
            }

            // Write image to file
            stbi_flip_vertically_on_write(true);                
            stbi_write_png(filename, SCREEN_WIDTH, SCREEN_HEIGHT, 4, pixelData.data(), SCREEN_WIDTH * 4);
        };

        if(recordExperiment){
            cameraMode = 1;
            std::vector<float> traversalTimeVector;
            std::vector<float> prepTimeVector;
            std::vector<float> renderTimeVector;
            std::vector<uint32_t> numRenSplatsVector;
            char filename[64];

            for(int i = 0; i < cameraData.size(); i++){
                /* Color */
                selectedViewMode = 0;
                softwareRasterizer(i);
                traversalTimeVector.push_back(traversalTime);
                prepTimeVector.push_back(prepTime);
                renderTimeVector.push_back(renderTime);
                numRenSplatsVector.push_back(renderedSplats);
                snprintf(filename, 64, "renders/%05d.png", i);
                saveRenderRoutine(filename);
                /* Depth */
                selectedViewMode = 1;
                softwareRasterizer(i);
                snprintf(filename, 64, "renders/d%05d.png", i);
                saveRenderRoutine(filename);
            }
            snprintf(filename, 64, "renders/stats_%.2f_%d.txt", diagonalProjectionThreshold, useFrustumCulling);
            std::ofstream csv_out(filename);
            if(csv_out.is_open()){
                csv_out << "FrameIdx,RenderedSplats,TraversalTime,PrepTime,RenderTime" << std::endl;
                for(int i = 0; i < cameraData.size(); i++){
                    csv_out << i << "," << numRenSplatsVector[i] 
                                 << "," << traversalTimeVector[i]
                                 << "," << prepTimeVector[i]
                                 << "," << renderTimeVector[i]
                                 << std::endl;
                }
                csv_out.close();
            }
        }

        if(batchRender){
            cameraMode = 1;
            for(int i = 0; i < cameraData.size(); i+=10){
                char filename[64];
                /* Color */
                selectedViewMode = 0;
                softwareRasterizer(i);
                snprintf(filename, 64, "renders/%05d.png", i);
                saveRenderRoutine(filename);
                /* Depth */
                selectedViewMode = 1;
                softwareRasterizer(i);
                snprintf(filename, 64, "renders/d%05d.png", i);
                saveRenderRoutine(filename);
            }
        }
        else{
            softwareRasterizer(cameraIndex);
        }

        /* Build ImGui interface */
        buildInterface();

        if(saveRender){
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);

            std::ostringstream oss;
            oss << "renders/output_" << std::put_time(&tm, "%d-%m %H-%M-%S")<<".png";
            auto str = oss.str();
            saveRenderRoutine(str.c_str());
        }

        // diagonalProjectionThreshold = 300.0f;

        /* Unmap the OpenGL resources */
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        /* Draw a quad which covers the entire screen */

        glBindTexture(GL_TEXTURE_2D, texId);
        glEnable(GL_TEXTURE_2D);

        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glTexCoord2f(0.0f, 0.0f);   glVertex3f(-1.0f, -1.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f);   glVertex3f( 1.0f, -1.0f, 0.0f);
        glTexCoord2f(1.0f, 1.0f);   glVertex3f( 1.0f,  1.0f, 0.0f);
        glTexCoord2f(0.0f, 1.0f);   glVertex3f(-1.0f,  1.0f, 0.0f);
        glEnd();

        /* Unbind the texture and PBO */
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        renderInterface();

        /* Swap buffers and handle GLFW events */
        glfwSwapBuffers(window);
        glfwPollEvents();

        /* Compute and display FPS every set number of frames */
        currentFPSIndex++;
        if(currentFPSIndex == FPS_COUNTER_REFRESH){
            end = std::chrono::steady_clock::now();
            int milisCount = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            char title[256];
            sprintf(title, "AccelerateGS [FPS: %f]", 1000.0f / milisCount * FPS_COUNTER_REFRESH);
            glfwSetWindowTitle(window, title);
            begin = std::chrono::steady_clock::now();
            currentFPSIndex = 0;
        }
    }

    shutdownIMGui();

    /* Unmap resources and free allocated memory */
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteTextures(1, &texId);
    glDeleteBuffers(1, &pboId);
    cudaFree(d_sd);
    cudaFree(d_pbo_buffer);

    cudaFree(d_conic_opacity);
    cudaFree(d_rgb);
    cudaFree(d_image_point);
    cudaFree(d_radius);
    cudaFree(d_depth);
    cudaFree(d_overlap);
    cudaFree(d_overlap_sums);
    cudaFree(d_cov3ds);
    cudaFree(d_renderMask);

    checkCudaErrors(cudaFree(d_tile_range_min));
    checkCudaErrors(cudaFree(d_tile_range_max));

    checkCudaErrors(cudaFree(d_sort_keys_in));
    checkCudaErrors(cudaFree(d_sort_keys_out));
    checkCudaErrors(cudaFree(d_sort_ids_in));
    checkCudaErrors(cudaFree(d_sort_ids_out));
    checkCudaErrors(cudaFree(d_temp_storage));

    if(spacePartitioningRoot != nullptr) delete spacePartitioningRoot;
    delete [] imageData;
    free(renderMask);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}