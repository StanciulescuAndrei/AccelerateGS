cmake_minimum_required(VERSION 3.21)
project(AccelerateGS VERSION 0.1.0 LANGUAGES C CXX CUDA)
cmake_policy(SET CMP0072 NEW)

find_package(OpenGL REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(OpenGL_GL_PREFERENCE GLVND)

include_directories(
	${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
)

add_executable(AccelerateGS 
		src/main.cu 
		src/glad/glad.h
		src/glad/glad.c)

target_link_libraries(AccelerateGS
	${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
	glfw
	${GLEW_LIBRARY}
)

