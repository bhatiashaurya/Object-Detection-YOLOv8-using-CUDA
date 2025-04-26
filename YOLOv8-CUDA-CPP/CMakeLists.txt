cmake_minimum_required(VERSION 3.10)
project(YOLOv8_CUDA_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV path
find_package(OpenCV REQUIRED PATHS "C:/opencv/build")

# CUDA support
enable_language(CUDA)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

file(GLOB SOURCES "src/*.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
