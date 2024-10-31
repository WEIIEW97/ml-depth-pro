cmake_minimum_required(VERSION 3.10)

project(onnx_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (UNIX AND NOT APPLE)
    # set(Torch_DIR "/home/william/Codes/libtorch/share/cmake/Torch")
    set(onnx_runtime_dir "/home/william/Codes/onnxruntime-linux-x64-gpu-1.18.0")
    # adjust it accordingly
    set(MATLAB_ROOT "/usr/local/MATLAB/R2023b")
elseif(WIN32)
    set(onnx_runtime_dir "D:/william/codes/onnxruntime-win-x64-gpu-1.18.0")
    # adjust it accordingly
    set(MATLAB_ROOT "C:/Program Files/MATLAB/R2023b")
else()
    # for apple support
    set(onnx_runtime_dir "/home/william/Codes/onnxruntime-linux-x64-gpu-1.18.0")
    set(MATLAB_ROOT "/usr/local/MATLAB/R2023b")
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

include(CheckIncludeFile)
include(CheckIncludeFileCXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Matlab REQUIRED MX_LIBRARY ENG_LIBRARY MAT_LIBRARY MAIN_PROGRAM)

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# file(GLOB_RECURSE torch_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch/*.cpp)
file(GLOB_RECURSE onnx_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/onnx/*.cpp)
file(GLOB_RECURSE trt_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tensorrt/*.cpp)
file(GLOB shared_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
file(GLOB_RECURSE mex_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mex/*.cpp)

file(GLOB_RECURSE headers ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h)


# add_executable(${PROJECT_NAME} 
#     ${onnx_sources}
#     ${shared_sources}
#     ${mex_sources}
#     main.cpp
# )

add_library(${PROJECT_NAME} 
    SHARED
    ${onnx_sources}
    ${shared_sources}
    ${mex_sources}
)


# if (Matlab_FOUND)
#     message(STATUS "MATLAB root directory found: ${Matlab_ROOT_DIR}.")
#     matlab_add_mex(
#         NAME depth_pro_onnx_infer
#         SRC
#         ${onnx_sources}
#         ${shared_sources}
#         ${mex_sources}
#         LINK_TO ${OpenCV_LIBS} ${LIB_ONNX}
#         ${LIB_PROVIDER_CUDA_ONNX}
#         )

#     #build MEX application and put it into bin directory
#     set_target_properties( depth_pro_onnx_infer
#         PROPERTIES
#         COMPILE_FLAGS "-fvisibility=default"
#         ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
#         LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
#         RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
#         )

#     message(STATUS "Mex file can be found in ${PROJECT_SOURCE_DIR}/bin")
# else ()
#     message(WARNING "MATLAB root directory not found. Failed to build MATLAB bindings.")
#     set(BUILD_MATLAB_BINDINGS OFF)
# endif ()

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${onnx_runtime_dir}/include/onnxruntime/core/session/
    ${onnx_runtime_dir}/include/onnxruntime/core/providers/tensorrt/
    ${onnx_runtime_dir}/include
    ${Matlab_INCLUDE_DIRS}
)

if(WIN32)
    set(LIB_ONNX "onnxruntime.lib")
    set(LIB_PROVIDER_CUDA_ONNX "onnxruntime_providers_cuda.lib")
elseif(UNIX AND NOT APPLE)
    set(LIB_ONNX "libonnxruntime.so")
    set(LIB_PROVIDER_CUDA_ONNX "libonnxruntime_providers_cuda.so")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE 
    # ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    ${LIB_ONNX}
    ${LIB_PROVIDER_CUDA_ONNX}
    ${Matlab_LIBRARIES}
)

check_include_file_cxx(tensorrt_provider_factory.h HAVE_TENSORRT_PROVIDER_FACTORY_H)


if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
endif()

target_link_directories(${PROJECT_NAME}
        PRIVATE
        ${onnx_runtime_dir}/lib
)

# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET ${PROJECT_NAME}
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:${PROJECT_NAME}>)
# endif (MSVC)


if(WIN32)
    # ONNX Runtime libraries
    set(ONNX_RUNTIME_DLL_PATH "${onnx_runtime_dir}/lib/onnxruntime.dll")
    set(ONNX_RUNTIME_PROVIDER_CUDA_DLL_PATH "${onnx_runtime_dir}/lib/onnxruntime_providers_cuda.dll")

    # OpenCV DLLs path (modify as per your OpenCV installation)
    set(OPENCV_DLL_DIR "C:/lib/vcpkg/installed/x64-windows/bin")  # Adjust this path as needed

    # Copy ONNX Runtime DLLs
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${ONNX_RUNTIME_DLL_PATH}
                       $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${ONNX_RUNTIME_PROVIDER_CUDA_DLL_PATH}
                       $<TARGET_FILE_DIR:${PROJECT_NAME}>)

    # Copy OpenCV DLLs
    file(GLOB OPENCV_DLLS "${OPENCV_DLL_DIR}/*opencv*.dll")
    foreach(DLL ${OPENCV_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           ${DLL}
                           $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endforeach()
endif()

if(WIN32)
    set(MEX_SUFFIX ".mexw64")
elseif(UNIX AND NOT APPLE)
    set(MEX_SUFFIX ".mexa64")
elseif(APPLE)
    set(MEX_SUFFIX ".mexmaci64")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ${MEX_SUFFIX})
set_target_properties(${PROJECT_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
)