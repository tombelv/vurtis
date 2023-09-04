cmake_minimum_required(VERSION 3.1)
project(vurtis)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")


add_library(${PROJECT_NAME} INTERFACE)

find_package(OsqpEigen REQUIRED)
find_package(proxsuite REQUIRED)
find_package(Eigen3 REQUIRED)

target_include_directories(${PROJECT_NAME} INTERFACE include/)

target_link_libraries(${PROJECT_NAME} INTERFACE OsqpEigen::OsqpEigen proxsuite::proxsuite-vectorized)

