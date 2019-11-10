cmake_minimum_required(VERSION 3.15.0)

project(rotation)

find_package(Ceres REQUIRED)
message("CERES_LIBRARIES = ${CERES_LIBRARIES}")
include_directories("C:/dev/eigen-eigen-5a0156e40feb")


include_directories(${CERES_INCLUDE_DIRS})
add_executable(rotation main.cpp tinyply.h )  
target_link_libraries(rotation ceres)

