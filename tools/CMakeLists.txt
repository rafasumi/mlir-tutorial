set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sblp-opt
  sblp-opt.cpp
)

llvm_update_compile_flags(sblp-opt)

target_link_libraries(sblp-opt
  PRIVATE

  SBLPTransforms

  # MLIR dependencies
  MLIROptLib
  ${dialect_libs}
  ${conversion_libs}
)

target_include_directories(sblp-opt
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

mlir_check_link_libraries(sblp-opt)
