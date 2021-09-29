cmake_minimum_required(VERSION 3.1)
project(vurtis)

set(TargetName "simple_example")

set (CMAKE_CXX_STANDARD 17)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(OsqpEigen REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(${TargetName} src/${TargetName}.cpp)
target_link_libraries(${TargetName} OsqpEigen::OsqpEigen)

