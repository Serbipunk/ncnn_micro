cmake_minimum_required(VERSION 3.16)

project(ncnn_model_MICRO)

set (CMAKE_CXX_STANDARD 17)
add_definitions(-fno-rtti)
# add_definitions(-frtti)

## 3rd parties
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -g")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -g ")
endif()

find_package(glog REQUIRED)
#find_package(rapidjson REQUIRED)  # todo

## ncnn shared library
SET(NCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ncnn-shared)
SET(NCNN_LIBS ${NCNN_DIR}/lib/libncnn.so)

set(CMAKE_BUILD_TYPE release)
add_definitions(-fPIC)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${NCNN_DIR}/include/
                    ${NCNN_DIR}/src/)

SET(SRC tool/ncnnm_disassembler/main.cc)
add_executable(ncnnm_disassembler ${SRC})
TARGET_LINK_LIBRARIES (ncnnm_disassembler 
                       ${NCNN_LIBS}
                       glog::glog
                       )
