#pragma once

#include "../../../libs/imgui/imgui.h"
#include "../../../libs/imgui/backends/imgui_impl_glfw.h"
#include "../../../libs/imgui/backends/imgui_impl_opengl3.h"
#include <string>

#define MIN_BVH_LEVEL 13
#define MAX_BVH_LEVEL 32

#define MIN_OCTREE_LEVEL 13
#define MAX_OCTREE_LEVEL 21

#define MIN_HYBRID_LEVEL octreeLevel
#define MAX_HYBRID_LEVEL 32

// #define MIN_BVH_LEVEL 4
// #define MAX_BVH_LEVEL 20

// #define MIN_OCTREE_LEVEL 4
// #define MAX_OCTREE_LEVEL 8

// #define MIN_HYBRID_LEVEL octreeLevel
// #define MAX_HYBRID_LEVEL 23

#define RA_SIZE 64

struct RenderConfig{
    std::string scene;
    std::string structure;
    std::string representative;
    std::string clustering;
    int numClusterFeatures;
    int spectralClusteringThreshold;
    float dbscan_epsilon;
    int nClusters;
    int octreeLevel;
};

class RollingAverage{
    private:
    float values[RA_SIZE];
    int ptr = 0;
    public:
    RollingAverage(){
        for(int i = 0; i < RA_SIZE; i++){
            values[i] = 0.0f;
        }
    }
    void insert(float value){
        values[ptr] = value;
        ptr = (ptr + 1) % RA_SIZE;
    }
    float get(){
        float sum = 0.0f;
        for(int i = 0; i < RA_SIZE; i++){
            sum += values[i];
        }
        return sum / RA_SIZE;
    }
};

// #define MIN_HYBRID_LEVEL 6
// #define MAX_HYBRID_LEVEL 15

float fovy = M_PI / 2.0f;
float fovx = M_PI / 2.0f * 16 / 9;
int selectedViewMode = 0;
int renderPrimitive = 0;
int renderLevel = MAX_HYBRID_LEVEL;
int cameraIndex = 158;
int cameraMode = 1;
int autoLevel = 0;

int saveRender = 0;
int batchRender = 0;
int recordExperiment = 0;

int renderedSplats = 0;
int duplicatedSplats = 0;
int maxSplats = 0;
int numCameraPositions = 2;
bool useFrustumCulling = 1;

float traversalTime = 0.0f;
float renderTime = 0.0f;
float prepTime = 0.0f;

int octreeLevel = 12;

RollingAverage avgTraversal;
RollingAverage avgRender;
RollingAverage avgPrep;

float diagonalProjectionThreshold = 0.0f;

RenderConfig renderConfig;

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

    int render_low_limit, render_high_limit;

    if(renderConfig.structure == std::string("octree")){
        render_low_limit = MIN_OCTREE_LEVEL;
        render_high_limit = MAX_OCTREE_LEVEL;
    }
    else if(renderConfig.structure == std::string("bvh")){
        render_low_limit = MIN_BVH_LEVEL; 
        render_high_limit = MAX_BVH_LEVEL;
    }
    else if(renderConfig.structure == std::string("hybrid")){
        render_low_limit = MIN_HYBRID_LEVEL; 
        render_high_limit = MAX_HYBRID_LEVEL;
    }

    ImGui::SliderInt("Render Level", &renderLevel, render_low_limit, render_high_limit + 1);
    ImGui::SliderFloat("LOD render bias", &diagonalProjectionThreshold, 0.0f, 300.0f);

    ImGui::Checkbox("Use frustum culling ", &useFrustumCulling);

    ImGui::Text("Traversal time: %.3f", avgTraversal.get()); 
    ImGui::Text("Prep time:      %.3f", avgPrep.get()); 
    ImGui::Text("Render time:    %.3f", avgRender.get()); 

    ImGui::RadioButton("Free camera", &cameraMode, 0); ImGui::SameLine();
    ImGui::RadioButton("COLMAP Camera", &cameraMode, 1);

    ImGui::SliderInt("Camera index", &cameraIndex, 0, numCameraPositions - 1);

    ImGui::Text("# rendered splats: %d", renderedSplats); 
    ImGui::Text("# duplicated splats: %d", duplicatedSplats); 
    ImGui::Text("# max splats: %d", maxSplats); 

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
    ImGui::SameLine();
    if(ImGui::Button("Record experiment")){
        recordExperiment = 1;
    }
    else{
        recordExperiment = 0;
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