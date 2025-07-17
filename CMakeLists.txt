cmake_minimum_required(VERSION 3.20.0)
project(MLIRTutorial)

set(CMAKE_CXX_STANDARD 17)

#===============================================================================
# LLVM Configuration
#===============================================================================
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

#===============================================================================
# Target Configuration
#===============================================================================
add_executable(matmul-opt
  matmul-opt.cpp
)

target_link_libraries(matmul-opt
  PRIVATE
  MLIROptLib
  MLIRArithDialect
)
