cmake_minimum_required(VERSION 2.8)

project(ncnn_model_MICRO)

set (CMAKE_CXX_STANDARD 17)
add_definitions(-fno-rtti)
# add_definitions(-frtti)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -g")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -g ")
endif()

find_package( OpenCV REQUIRED )    # opencv 3
message("OpenCV_DIR: ${OpenCV_DIR}")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS: ${OpenCV_LIBS}")

SET(NCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ncnn-shared)
SET(NCNN_LIBS ${NCNN_DIR}/lib/libncnn.so)

set(CMAKE_BUILD_TYPE release)
add_definitions(-fPIC)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${NCNN_DIR}/include/
                    ${NCNN_DIR}/src/)

SET(SRC tool/ncnnm_disassembler/main.cc)
add_executable(ncnnm_disassembler ${SRC})
TARGET_LINK_LIBRARIES (ncnnm_disassembler 
                       ${OpenCV_LIBS} 
                       ${NCNN_LIBS}
                       glog)
