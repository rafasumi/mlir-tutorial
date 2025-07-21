add_mlir_library(SBLPTransforms
    StrengthReduction.cpp

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRMathDialect
      MLIRFuncDialect
)

target_include_directories(SBLPTransforms
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
