cmake_minimum_required(VERSION 3.17)
project(Segmentation)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3.4 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)
find_package(OpenCV REQUIRED)
find_package(CUDA)
include_directories(include)

set(CMAKE_PREFIX_PATH
        /usr/local/libtorch)
set(Torch_DIR libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)


FILE(GLOB ALL_SOURCES "*.cpp" "./src/*.cpp"  "./src/architectures/*.cpp"  "./src/backbones/*.cpp"  "./src/utils/*.cpp" )
add_executable(Segmentation src/main.cpp ${ALL_SOURCES})

target_link_libraries(Segmentation ${OpenCV_LIBS})
target_link_libraries(Segmentation ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(Segmentation
        ${TORCH_LIBRARIES}
        )