cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(AccelerateGS VERSION 0.1.0 LANGUAGES CXX C CUDA)
cmake_policy(SET CMP0072 NEW)

list(APPEND CMAKE_PREFIX_PATH "/home/andrei/Documents/libs/libtorch")

find_package(OpenGL REQUIRED)

find_package(OpenCL REQUIRED)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

find_package(OpenMP)

find_package(Threads REQUIRED)

find_package(nanoflann)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CUDA_ARCHITECTURES 86)
set(OpenGL_GL_PREFERENCE GLVND)

set(IMGUI_PATH ../../libs/imgui)

set(STB_PATH ../../libs/stb)

include_directories(
	${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
	../../libs/imgui/
	../../libs/imgui/backends/
	../../libs/nlohmann/
	${STB_PATH}
)

add_compile_definitions("RELEASE=$<CONFIG:Release>")

add_executable(AccelerateGS 
		src/main.cu 
		src/glad/glad.h
		src/glad/glad.c
		src/PLYReader.h
		src/GUIManager.h
		src/SpacePartitioningBase.h
		src/GaussianOctree.h
		src/GaussianBVH.h
		src/HybridVH.h
		src/raster_helper.cuh
		src/CameraLoader.h
		src/PointClustering.h
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
		${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp
		../../libs/nlohmann/json.hpp
		${STB_PATH}/stb_image_write.h
		)

target_link_libraries(AccelerateGS
	${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
	glfw
	${GLEW_LIBRARY}
	Eigen3::Eigen
	Threads::Threads
	nanoflann::nanoflann
	tbb
	tbbmalloc
	pthread
	OpenCL
)

set_property(TARGET AccelerateGS PROPERTY CXX_STANDARD 17)

