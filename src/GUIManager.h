#pragma once

#include "../../../libs/imgui/imgui.h"
#include "../../../libs/imgui/backends/imgui_impl_glfw.h"
#include "../../../libs/imgui/backends/imgui_impl_opengl3.h"

float fovy = M_PI / 2.0f;
int selectedViewMode = 0;
int renderPrimitive = 0;

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

    ImGui::Text("Placeholder for future reference...");  
    ImGui::SliderFloat("FOV", &fovy, 0.5f, 2.7f, "%.2f");

    const char * viewMode[]{"Colors", "Depth", "Covariance"};
    ImGui::Combo("View mode", &selectedViewMode, viewMode, 3);
    ImGui::RadioButton("Splats", &renderPrimitive, 0); ImGui::SameLine();
    ImGui::RadioButton("Points", &renderPrimitive, 1);

    ImGui::End();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void shutdownIMGui(){
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}