if [ -z "$LLVM_BUILD_DIR" ]; then
    >&2 echo "'LLVM_BUILD_DIR' variable is not set"
    exit 1;
fi

if [ ! -d "$LLVM_BUILD_DIR" ]; then
    >&2 echo "'LLVM_BUILD_DIR' is not a valid directory"
    exit 1;
fi
