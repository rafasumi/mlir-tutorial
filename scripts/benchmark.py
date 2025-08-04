import ctypes
import numpy as np
import time


class MemRef2D(ctypes.Structure):
    """Structure matching 2D memref MLIR descriptor"""

    _fields_ = [
        ("allocated", ctypes.c_void_p),  # Allocated pointer
        ("aligned", ctypes.c_void_p),  # Aligned pointer
        ("offset", ctypes.c_longlong),  # Offset in elements
        ("shape", ctypes.c_longlong * 2),  # Array shape (2D)
        ("stride", ctypes.c_longlong * 2),  # Strides in elements
    ]

    def get_arg_types(self):
        """Get types of memref fields that will be passed as arguments"""
        arg_types = []
        for _, field_type in self._fields_:
            if issubclass(field_type, ctypes.Array):
                arg_types.extend([field_type._type_] * field_type._length_)
            else:
                arg_types.append(field_type)
        return arg_types

    def get_as_arg_list(self):
        """Get memref fields that will be passed as arguments"""
        args = []
        for field_name, field_type in self._fields_:
            value = getattr(self, field_name)
            if issubclass(field_type, ctypes.Array):
                args.extend(value)
            else:
                args.append(value)
        return args


def numpy_to_memref2d(arr):
    """Convert a 2D NumPy array to a MemRef descriptor"""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    desc = MemRef2D()
    desc.allocated = arr.ctypes.data_as(ctypes.c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    desc.shape[0] = arr.shape[0]
    desc.shape[1] = arr.shape[1]
    desc.stride[0] = arr.strides[0] // arr.itemsize
    desc.stride[1] = arr.strides[1] // arr.itemsize

    return desc


def run_matmul(a_matrix, b_matrix, c_matrix, dylib_path):
    """
    Run matrix multiplication for a compiled MLIR module and measure
    execution time
    """

    # Load matmul module compiled from MLIR
    module = ctypes.CDLL(dylib_path)

    # Prepare MemRef descriptors
    a_memref = numpy_to_memref2d(a_matrix)
    b_memref = numpy_to_memref2d(b_matrix)
    c_memref = numpy_to_memref2d(c_matrix)

    # Set function argument types
    module.matmul.argtypes = [
        *a_memref.get_arg_types(),
        *b_memref.get_arg_types(),
        *c_memref.get_arg_types(),
    ]
    module.matmul.restype = None

    # Call the function and measure execution time in nanoseconds
    start = time.perf_counter_ns()
    module.matmul(
        *a_memref.get_as_arg_list(),
        *b_memref.get_as_arg_list(),
        *c_memref.get_as_arg_list(),
    )
    end = time.perf_counter_ns()

    return end - start


def print_result(opt, exec_time, max_len):
    print(f"{opt:>{max_len}}: {exec_time:>15}ns")


def print_header(dim1, dim2, dim3, dtype):
    title = f"Benchmark MLIR ({dim1}x{dim2})*({dim2}x{dim3}) {dtype.__name__} matrix multiplication"
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main():
    dim1 = 1024
    dim2 = 256
    dim3 = 1024
    dtype = np.float32

    print_header(dim1, dim2, dim3, dtype)

    # Get random inputs
    a_matrix = np.random.rand(dim1, dim2).astype(dtype)
    b_matrix = np.random.rand(dim2, dim3).astype(dtype)

    # Expected result of matrix multiplication
    desired = np.matmul(a_matrix, b_matrix)

    # Run matmul with each optimization level, while verifying performance
    opts = ["baseline", "unroll"]
    max_len = max(len(opt) for opt in opts)

    for opt in opts:
        dylib = f"./tmp/libmatmul{opt}.dylib"
        c_matrix = np.zeros((dim1, dim3), dtype=dtype)
        exec_time = run_matmul(a_matrix, b_matrix, c_matrix, dylib)

        # Check if result is correct
        np.testing.assert_almost_equal(c_matrix, desired, decimal=4)
        print_result(opt, exec_time, max_len)


if __name__ == "__main__":
    main()
