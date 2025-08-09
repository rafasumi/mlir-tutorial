set(LIT_CFG_PY "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${LIT_CFG_PY}
)

set(SBLP_TEST_DEPENDS
  FileCheck
  sblp-opt
)

add_lit_testsuite(check-sblp "Running SBLP Tutorial regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SBLP_TEST_DEPENDS}
)

set_target_properties(check-sblp PROPERTIES FOLDER "Tests")
