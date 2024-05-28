#pragma once

#include "../../../libs/imgui/imgui.h"
#include "../../../libs/imgui/backends/imgui_impl_glfw.h"
#include "../../../libs/imgui/backends/imgui_impl_opengl3.h"

#define MAX_BVH_LEVEL 23
#define MIN_BVH_RESOLUTION MAX_BVH_LEVEL - 7

#define MAX_OCTREE_LEVEL 15
#define MIN_RESOLUTION MAX_OCTREE_LEVEL - 4

// #define INRIA_CLUSTER

float fovy = M_PI / 2.0f;
float fovx = M_PI / 2.0f * 16 / 9;
int selectedViewMode = 0;
int renderPrimitive = 0;
int renderLevel = MIN_BVH_RESOLUTION;
int cameraIndex = 0;
int cameraMode = 0;
int autoLevel = 0;

int saveRender = 0;
int batchRender = 0;

int renderedSplats = 0;
int numCameraPositions = 2;

float diagonalProjectionThreshold = 10.0f;

void setupIMGui(GLFWwindow** window){
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(*window, true);
    ImGui_ImplOpenGL3_Init();
}

void buildInterface(){
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Render Settings");

    ImGui::SliderFloat("FOV", &fovy, 0.5f, 2.7f, "%.2f");

    const char * viewMode[]{"Colors", "Depth", "Covariance"};
    ImGui::Combo("View mode", &selectedViewMode, viewMode, 3);
    ImGui::RadioButton("Splats", &renderPrimitive, 0); ImGui::SameLine();
    ImGui::RadioButton("Points", &renderPrimitive, 1);

    ImGui::Text("LOD selection: "); ImGui::SameLine();
    ImGui::RadioButton("Auto", &autoLevel, 1); ImGui::SameLine();
    ImGui::RadioButton("Manual", &autoLevel, 0);

    ImGui::SliderInt("Render Level", &renderLevel, MIN_BVH_RESOLUTION, MAX_BVH_LEVEL + 1);
    ImGui::SliderFloat("LOD render bias", &diagonalProjectionThreshold, 10.0f, 100.0f);

    ImGui::RadioButton("Free camera", &cameraMode, 0); ImGui::SameLine();
    ImGui::RadioButton("COLMAP Camera", &cameraMode, 1);

    ImGui::SliderInt("Camera index", &cameraIndex, 0, numCameraPositions - 1);

    ImGui::Text("# rendered splats: %d", renderedSplats); 

    if(ImGui::Button("Capture render")){
        saveRender = 1;
    }
    else{
        saveRender = 0;
    }
    ImGui::SameLine();
    if(ImGui::Button("Batch render")){
        batchRender = 1;
    }
    else{
        batchRender = 0;
    }

    ImGui::End();
}

void buildLoadingInterface(float progress){
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // Create a window called "Progress" and append into it.
    ImGui::Begin("Building space partitioning");

    // Display the progress bar
    ImGui::ProgressBar(progress, ImVec2(0.0f,0.0f));
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::Text("Progress %.0f%%", progress*100.0f);

    ImGui::End();
}

void renderInterface(){
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void shutdownIMGui(){
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}