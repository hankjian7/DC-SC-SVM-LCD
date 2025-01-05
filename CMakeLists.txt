# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_raft)

project(test_raft LANGUAGES CXX CUDA)
# set Eigen3
find_package(Eigen3 3.3 REQUIRED)
# set yaml-cpp
find_package(yaml-cpp REQUIRED)
# set Boost
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
include_directories(${Boost_INCLUDE_DIRS})
# set OpenMP
# find_package(OpenMP REQUIRED)
# Find OpenCV package
find_package(OpenCV REQUIRED)
# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)

# -------------- compile tasks ----------------- #
# Include all header files
include_directories(header)
# include_directories(original_version/header)
# List all source files
# set(Original
#     src/BruteForceKNN_raft.cu
#     src/MinHeapWithTracking.cpp
#     src/Hamming.cpp
#     src/Matching.cpp
#     src/svm.cpp
#     src/calculate.cpp
#     src/CodebookWithSpare.cpp
#     original_version/src/InvertedFile.cpp
#     original_version/src/LCDEngineOriginal.cpp
# )
set(MainSrcs
    src/svm.cpp
    src/calculate.cpp
    src/Matching.cpp
    src/BruteForceKNN_raft.cu
    src/MinHeapWithTracking.cpp
    src/Hamming.cpp
    src/Searcher.cpp
    src/CodebookWithSpare.cpp
    src/LCDEngine.cpp
)
add_executable(LCDEngine ${MainSrcs})
target_link_libraries(LCDEngine PUBLIC yaml-cpp raft::raft raft::compiled Eigen3::Eigen ${Boost_LIBRARIES} ${OpenCV_LIBS})