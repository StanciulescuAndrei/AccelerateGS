#pragma once

#include "../../../libs/imgui/imgui.h"
#include "../../../libs/imgui/backends/imgui_impl_glfw.h"
#include "../../../libs/imgui/backends/imgui_impl_opengl3.h"

float fovy = M_PI / 2.0f;

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

    ImGui::Begin("Render Settings");                          // Create a window called "Hello, world!" and append into it.

    ImGui::Text("Placeholder for future reference...");  
    ImGui::SliderFloat("FOV", &fovy, 0.5f, 1.5f, "%.2f");

    ImGui::End();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void shutdownIMGui(){
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}