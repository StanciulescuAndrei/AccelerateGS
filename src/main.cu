#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <iostream>

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <chrono>

#include "render.cuh"

#include "cuda_common/helper_cuda.h"


#define N 100000000
#define MAX_ERR 1e-6

const int SCREEN_WIDTH = 960;
const int SCREEN_HEIGHT = 960;



const int FPS_COUNTER_REFRESH = 60;

glm::vec3 cameraPosition = glm::vec3(0.0f);
const float movement_step = 0.1f;

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}
 
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if(key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += glm::vec3(0.0f, movement_step, 0.0f);
    }
    if(key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += glm::vec3(0.0f, -movement_step, 0.0f);
    }
    if(key == GLFW_KEY_Z && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += glm::vec3(movement_step, 0.0f, 0.0f);
    }
    if(key == GLFW_KEY_X && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += glm::vec3(-movement_step, 0.0f, 0.0f);
    }
    if(key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += glm::vec3(0.0f, 0.0f, -movement_step);
    }
    if(key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS)){
        cameraPosition += glm::vec3(0.0f, 0.0f, movement_step);
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
}

int main(){
    GLFWwindow* window;

    initGLContextAndWindow(&window);

    /* Load splat scene data from file */
    SplatData * sd;
    int num_elements = 0;
    int res = loadSplatData("../../models/train/point_cloud/iteration_30000/point_cloud.ply", &sd, &num_elements);

    num_elements = 1000;

    /* Allocate and send splat data to GPU memory */
    SplatData * d_sd;
    checkCudaErrors(cudaMalloc(&d_sd, sizeof(SplatData) * num_elements));
    assert(d_sd != NULL);
    checkCudaErrors(cudaMemcpy((void*)d_sd, (void*) sd, sizeof(SplatData) * num_elements, cudaMemcpyHostToDevice));

    /* Allocate additional data buffers */
    float4 * d_conic_opacity;
    float3 * d_rgb;
    float2 * d_image_point;
    int * d_radius;
    float * d_depth;
    int * d_overlap;

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

    /* Set up resources for texture writing */
    GLuint pboId;
    GLuint texId;
    GLfloat * imageData = new GLfloat[SCREEN_HEIGHT * SCREEN_WIDTH * 4];

    struct cudaGraphicsResource * cuda_pbo_resource;
    void * d_pbo_buffer = NULL;

    // Initialize the texture
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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

    glPixelStorei(GL_UNPACK_ALIGNMENT, 16);      // 4-byte pixel alignment

    glClearColor(0, 0, 0, 0);                   // background color
    glClearStencil(0);                          // clear stencil buffer
    glClearDepth(1.0f);                         // 0 is near, 1 is far
    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

    /* Very basic FPS metrics */
    int currentFPSIndex = 0;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    begin = std::chrono::steady_clock::now();

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

        /* --------- RENDERING ------------*/
        glm::mat4 modelview = glm::translate(glm::mat4(1.0f), cameraPosition);
        glm::mat4 perspective = glm::perspective(90.0f, 16.0f/9.0f, 1.0f, 200.0f);

        /* Call the main CUDA render kernel */
        dim3 block(BLOCK_X, BLOCK_Y, 1); // One thread per pixel!
        dim3 grid(SCREEN_HEIGHT / BLOCK_X + 1, SCREEN_WIDTH / BLOCK_Y + 1, 1);

        preprocessGaussians<<<num_elements / 1024 + 1, 1024>>>(num_elements, d_sd, perspective, modelview, d_conic_opacity, d_rgb, d_image_point, d_radius, d_depth, d_overlap, SCREEN_HEIGHT, SCREEN_WIDTH, grid);
        checkCudaErrors(cudaDeviceSynchronize());
        cumulativeSum<<<1,1>>>(num_elements, d_overlap);
        checkCudaErrors(cudaDeviceSynchronize());

        int totalDuplicateGaussians = 0;
        checkCudaErrors(cudaMemcpy(&totalDuplicateGaussians, d_overlap + num_elements - 1, sizeof(int), cudaMemcpyDeviceToHost));

        /* Now create an array to keep the tile id and depth (32 bits + 32 bits) */ 
        uint64_t * d_sort_keys;
        checkCudaErrors(cudaMalloc(&d_sort_keys, sizeof(uint64_t) * totalDuplicateGaussians));

        /* Populate sorting keys array */

        // debugInfo<<<1, 1>>>(num_elements, d_sd, perspective, modelview, d_conic_opacity, d_rgb, d_image_point, d_radius, d_depth, d_overlap, SCREEN_HEIGHT, SCREEN_WIDTH, grid);
        // checkCudaErrors(cudaDeviceSynchronize());
        render<<<grid, block>>>(1000, d_sd, perspective, modelview, d_conic_opacity, d_rgb, d_image_point, d_radius, d_depth, d_overlap, SCREEN_HEIGHT, SCREEN_WIDTH, grid, dataPointer);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_sort_keys));

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

    delete [] imageData;

    return 0;
}