cmake_minimum_required(VERSION 3.20.0)
project(SBLPMLIRTutorial)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#===============================================================================
# LLVM config
#===============================================================================
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_LLVM_DIR}")

include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

#===============================================================================
# Project config
#===============================================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
