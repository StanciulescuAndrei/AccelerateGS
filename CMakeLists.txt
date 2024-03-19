cmake_minimum_required(VERSION 3.16)
project(AccelerateGS VERSION 0.1.0 LANGUAGES C CXX CUDA)
cmake_policy(SET CMP0072 NEW)

find_package(OpenGL REQUIRED)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(OpenGL_GL_PREFERENCE GLVND)

set(IMGUI_PATH /home/andrei/Documents/libs/imgui)

include_directories(
	${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
	../../libs/imgui/
	../../libs/imgui/backends/
)

add_executable(AccelerateGS 
		src/main.cu 
		src/glad/glad.h
		src/glad/glad.c
		src/PLYReader.h
		src/GUIManager.h
		src/GaussianOctree.h
		${IMGUI_PATH}/imgui.h
		${IMGUI_PATH}/imgui.cpp
		${IMGUI_PATH}/imconfig.h
		${IMGUI_PATH}/imgui_demo.cpp
		${IMGUI_PATH}/imgui_draw.cpp
		${IMGUI_PATH}/imgui_internal.h
		${IMGUI_PATH}/imstb_rectpack.h
		${IMGUI_PATH}/imstb_textedit.h
		${IMGUI_PATH}/imstb_truetype.h
		${IMGUI_PATH}/imgui_tables.cpp
		${IMGUI_PATH}/imgui_widgets.cpp
		${IMGUI_PATH}/backends/imgui_impl_glfw.h
		${IMGUI_PATH}/backends/imgui_impl_glfw.cpp
		${IMGUI_PATH}/backends/imgui_impl_opengl3.h
		${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp)

target_link_libraries(AccelerateGS
	${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
	glfw
	${GLEW_LIBRARY}
	Eigen3::Eigen
)

