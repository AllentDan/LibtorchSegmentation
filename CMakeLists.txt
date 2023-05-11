cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)

set(PROJECT_VERSION 1.0.0)

include(GNUInstallDirs)

project(LibTorchSegmentation VERSION ${PROJECT_VERSION}
    DESCRIPTION "Image Segmentation library based on LibTorch")


# First of all set your libtorch path.
set(Torch_DIR $ENV{Torch_DIR}/share/cmake/Torch)

find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

# At this point, OpenCV should be already installed
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV VERSION " ${OpenCV_VERSION})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)
find_package(OpenCV REQUIRED)


FILE(GLOB ALL_SOURCES
    "src/*.cpp"
    "src/architectures/*.cpp"
    "src/backbones/*.cpp"
    "src/utils/*.cpp"
)


add_library(segmentation ${ALL_SOURCES})

configure_file(segmentation.pc.in segmentation.pc @ONLY)

set(ROOT_PUBLIC_HEADERS
    "src/SegDataset.h"
    "src/Segmentor.h"
)
 set_target_properties(segmentation PROPERTIES
     PUBLIC_HEADER "${ROOT_PUBLIC_HEADERS}")

install(TARGETS segmentation
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/segmentation)

set(ARCHITECTURES_PUBLIC_HEADERS
    "src/architectures/DeepLabDecoder.h"
    "src/architectures/DeepLab.h"
    "src/architectures/FPNDecoder.h"
    "src/architectures/FPN.h"
    "src/architectures/LinknetDecoder.h"
    "src/architectures/LinkNet.h"
    "src/architectures/PANDecoder.h"
    "src/architectures/PAN.h"
    "src/architectures/PSPNetDecoder.h"
    "src/architectures/PSPNet.h"
    "src/architectures/UNetDecoder.h"
    "src/architectures/UNet.h"
)
 set_target_properties(segmentation PROPERTIES
    PUBLIC_HEADER "${ARCHITECTURES_PUBLIC_HEADERS}"
)

install(TARGETS segmentation
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/segmentation/architectures)


set(BACKBONES_PUBLIC_HEADERS
    "src/backbones/ResNet.h"
    "src/backbones/VGG.h"
)
 set_target_properties(segmentation PROPERTIES
    PUBLIC_HEADER "${BACKBONES_PUBLIC_HEADERS}"
)

install(TARGETS segmentation
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/segmentation/backbones)

set(UTILS_PUBLIC_HEADERS
    "src/utils/Augmentations.h"
    "src/utils/_dirent.h"
    "src/utils/InterFace.h"
    "src/utils/loss.h"
    "src/utils/readfile.h"
    "src/utils/util.h"
    "src/utils/json.hpp"
)

set_target_properties(segmentation PROPERTIES
    PUBLIC_HEADER "${UTILS_PUBLIC_HEADERS}"
)

install(TARGETS segmentation
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/segmentation/utils)

#install(TARGETS segmentation
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#)

install(FILES ${CMAKE_BINARY_DIR}/segmentation.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)


include(CMakePackageConfigHelpers)

configure_package_config_file(
    "Config.cmake.in"
    "SegmentationConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Segmentation
    PATH_VARS
    CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SegmentationConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

### Install Config and ConfigVersion files
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/SegmentationConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SegmentationConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Segmentation"
)


set_target_properties(segmentation PROPERTIES VERSION ${PROJECT_VERSION})

set(ALL_LIBS
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cudart_static_LIBRARY}
)

target_link_libraries(segmentation ${ALL_LIBS})


