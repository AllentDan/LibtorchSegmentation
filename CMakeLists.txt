cmake_minimum_required(VERSION 3.10)
project(LibtorchSegmentation)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)
find_package(OpenCV REQUIRED)
include_directories(include)

set(CMAKE_PREFIX_PATH "/home/SENSETIME/dongchunyu/dongchunyu/depends/libtorch181cu111")
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
add_executable(LibtorchSegmentation src/main.cpp ${ALL_SOURCES})

target_link_libraries(LibtorchSegmentation ${OpenCV_LIBS})
target_link_libraries(LibtorchSegmentation ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(LibtorchSegmentation
        ${TORCH_LIBRARIES}
        )