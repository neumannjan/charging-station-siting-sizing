cmake_minimum_required(VERSION 3.4...3.18)
project(fleet_electrification_cpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(cpp/pybind11)
pybind11_add_module(
  fleet_electrification_cpp
  cpp/src/log.cpp
  cpp/src/attempts.cpp
  cpp/src/simulation.cpp
  cpp/src/soft_assignment/graph.cpp
  cpp/src/soft_assignment/soft_assignment.cpp
  cpp/src/main.cpp
)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")

target_include_directories(fleet_electrification_cpp PRIVATE cpp/include cpp/parallel_hashmap/)
target_compile_features(fleet_electrification_cpp PUBLIC cxx_std_17)
