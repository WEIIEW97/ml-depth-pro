cmake_minimum_required(VERSION 3.10)

project(depth-pro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR "/home/william/Codes/libtorch/share/cmake/Torch")
set(onnx_runtime_dir "/home/william/Codes/onnxruntime-linux-x64-gpu-1.19.2")

include(CheckIncludeFile)
include(CheckIncludeFileCXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE torch_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch/*.cpp)
file(GLOB_RECURSE onnx_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/onnx/*.cpp)
file(GLOB_RECURSE trt_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tensorrt/*.cpp)
file(GLOB shared_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)

file(GLOB_RECURSE headers ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h)


add_executable(${PROJECT_NAME} 
    ${onnx_sources}
    ${shared_sources} 
    main.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${onnx_runtime_dir}/include/onnxruntime/core/session/
    ${onnx_runtime_dir}/include/onnxruntime/core/providers/tensorrt/
    ${onnx_runtime_dir}/include

)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE 
    # ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    libonnxruntime.so
)

check_include_file_cxx(tensorrt_provider_factory.h HAVE_TENSORRT_PROVIDER_FACTORY_H)


if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
endif()

target_link_directories(${PROJECT_NAME}
        PRIVATE
        ${onnx_runtime_dir}/lib
)



if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)