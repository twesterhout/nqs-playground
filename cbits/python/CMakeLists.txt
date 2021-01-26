find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# Python bindings
pybind11_add_module(_C MODULE NO_EXTRAS
    trim.cpp
    init.cpp
)
nqs_cbits_add_low_level_flags(_C)
target_compile_definitions(_C PUBLIC
    TORCH_API_INCLUDE_EXTENSION_H=1
    TORCH_EXTENSION_NAME="_C")
target_link_libraries(_C PRIVATE nqs ${TORCH_PYTHON_LIBRARY})
