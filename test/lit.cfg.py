import os

import lit.formats
from lit.llvm import llvm_config

config.name = "SBLP"

config.test_source_root = os.path.dirname(__file__)
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]

llvm_config.with_environment("PATH", config.bin_dir, append_path=True)
