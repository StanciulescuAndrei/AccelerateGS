cmake_minimum_required(VERSION 3.16)
project(AccelerateGS VERSION 0.1.0 LANGUAGES CXX C CUDA)
cmake_policy(SET CMP0072 NEW)

find_package(OpenGL REQUIRED)

find_package(OpenCL REQUIRED)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

find_package(OpenMP)

find_package(nanoflann)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")


set(CMAKE_CUDA_ARCHITECTURES 86)
set(OpenGL_GL_PREFERENCE GLVND)

set(IMGUI_PATH /home/andrei/Documents/libs/imgui)

set(STB_PATH /home/andrei/Documents/libs/stb)

include_directories(
	${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
	../../libs/imgui/
	../../libs/imgui/backends/
	../../libs/nlohmann/
	${STB_PATH}
	/opt/intel/oneapi/dal/2024.2/include/
	/opt/intel/oneapi/dal/2024.2/include/dal
)

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
		src/OctreeSerializer.cuh
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
	nanoflann::nanoflann
	/opt/intel/oneapi/dal/2024.2/lib/libonedal.a
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_dpc.a
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_parameters_dpc.a
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_sycl.a
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_parameters.so.2
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_core.so.2 
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_thread.so.2
	/opt/intel/oneapi/tbb/2021.12/lib/intel64/gcc4.8/libtbb.so.12 
	/opt/intel/oneapi/tbb/2021.12/lib/intel64/gcc4.8/libtbbmalloc.so.2 
	/opt/intel/oneapi/dal/2024.2/lib/libonedal.so.2 
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_parameters.so.2 
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_core.so.2 
	/opt/intel/oneapi/dal/2024.2/lib/libonedal_thread.so.2
	tbb
	tbbmalloc
	pthread
	sycl
	OpenCL
)

