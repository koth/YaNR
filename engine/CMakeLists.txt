cmake_minimum_required(VERSION 3.15)
project(nr_student_cpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories(include src)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_library(nr_cpu STATIC src/lanes.cpp src/engine.cpp)
target_compile_options(nr_cpu PRIVATE -O3)
target_link_libraries(nr_cpu OpenMP::OpenMP_CXX Threads::Threads)

add_executable(cpu_lanes tools/lanes_main.cpp)
target_link_libraries(cpu_lanes nr_cpu)

add_executable(cpu_engine tools/engine_main.cpp)
target_link_libraries(cpu_engine nr_cpu)

add_executable(hash_probe tools/hash_probe.cpp)
target_link_libraries(hash_probe nr_cpu)
