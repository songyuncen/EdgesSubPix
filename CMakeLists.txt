cmake_minimum_required(VERSION 3.0)
project(EdgesSubPix)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(EdgesSubPix EdgesSubPix.cpp EdgesSubPix.h)
target_link_libraries(EdgesSubPix ${OpenCV_LIBS})

add_executable(edge test_EdgesSubPix.cpp)
target_link_libraries(edge EdgesSubPix ${OpenCV_LIBS})


