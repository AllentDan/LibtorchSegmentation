include(GNUInstallDirs)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)

set(PROJECT_VERSION 1.0.0)


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

if(BUILD_SHARED)
    add_library(segmentation SHARED ${ALL_SOURCES})
    message(STATUS "Target shared library")
else()
    add_library(segmentation STATIC ${ALL_SOURCES})
    message(STATUS "Target static library")
endif()

configure_file(segmentation.pc.in segmentation.pc @ONLY)


install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/segmentation
        FILES_MATCHING
        PATTERN "*.h*"
)

install(TARGETS segmentation
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

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


