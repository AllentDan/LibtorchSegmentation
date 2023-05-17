cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

project(AppSegmentation)

set(Torch_DIR $ENV{Torch_DIR}/share/cmake/Torch)

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

if (Torch_FOUND)
    message(STATUS "Torch library found!")
else ()
    message(FATAL_ERROR "Could not locate Torch")
endif()

find_package(Segmentation REQUIRED)

if (Segmentation_FOUND)
    message(STATUS "Segmentation library found!")
else ()
    message(FATAL_ERROR "Could not locate Segmentation")
endif()

include_directories(
    ${SEGMENTATION_INCLUDE_DIRS}
)

set(ALL_LIBS
    segmentation
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cudart_static_LIBRARY}
    )

add_executable(train train.cpp)
target_link_libraries(train ${ALL_LIBS})

add_executable(resnet34 resnet34.cpp)
target_link_libraries(resnet34 ${ALL_LIBS})


