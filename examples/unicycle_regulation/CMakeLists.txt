cmake_minimum_required(VERSION 3.1)
project(example)

set(TargetName "regulation")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# add the main vurtis directory
add_subdirectory(../.. build)

add_executable(${TargetName} ${TargetName}.cpp)

target_link_libraries(${TargetName} vurtis)

