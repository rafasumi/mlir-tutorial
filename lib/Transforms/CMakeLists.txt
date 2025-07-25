add_mlir_library(SBLPTransforms
    Pipeline.cpp
    StrengthReduction.cpp

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRAffineDialect
      MLIRArithDialect
      MLIRLinalgDialect
      MLIRBufferizationDialect
      MLIRFuncDialect
)

target_include_directories(SBLPTransforms
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
