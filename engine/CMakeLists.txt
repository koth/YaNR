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

# AVX2 GEMM 微内核(gemm.cpp 单独开 -mavx2 -mfma,运行期 __builtin_cpu_supports
# 守卫回落标量;其余 TU 保持基线 ISA,便于 MSVC/老机器移植)。
option(NR_AVX2 "Build the AVX2 GEMM microkernel (x86-64 hosts)" ON)
if(NR_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(NR_AVX2_FLAGS "-mavx2;-mfma")
elseif(NR_AVX2 AND MSVC)
  set(NR_AVX2_FLAGS "/arch:AVX2")
else()
  set(NR_AVX2_FLAGS "")
endif()

add_library(nr_cpu STATIC src/lanes.cpp src/engine.cpp src/attn.cpp src/gemm.cpp)
target_compile_options(nr_cpu PRIVATE -O3)
if(NR_AVX2_FLAGS)
  # 只对数值内核 TU 开 AVX2(运行期 __builtin_cpu_supports 守卫);其余 TU 保持
  # 基线 ISA,便于 MSVC/老机器移植。
  set_source_files_properties(src/attn.cpp src/gemm.cpp PROPERTIES COMPILE_OPTIONS "${NR_AVX2_FLAGS}")
endif()
target_link_libraries(nr_cpu OpenMP::OpenMP_CXX Threads::Threads)

add_executable(cpu_lanes tools/lanes_main.cpp)
target_link_libraries(cpu_lanes nr_cpu)

add_executable(cpu_engine tools/engine_main.cpp)
target_link_libraries(cpu_engine nr_cpu)

add_executable(hash_probe tools/hash_probe.cpp)
target_link_libraries(hash_probe nr_cpu)

add_executable(gemm_bench tools/gemm_bench.cpp)
target_link_libraries(gemm_bench nr_cpu)
