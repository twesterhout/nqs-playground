# Determine if we have nvcc
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
else()
    message(WARNING "[nqs_cbits] Compiling without CUDA support")
endif()

# Library of CUDA specific kernels
if(CMAKE_CUDA_COMPILER)
    add_library(nqs_gpu OBJECT
        unpack.hpp
        unpack.cu
    )
    nqs_cbits_add_low_level_flags(nqs_gpu)
    target_compile_options(nqs_gpu PRIVATE --compiler-options "-W -Wall -Wextra")
    target_link_libraries(nqs_gpu PUBLIC gsl::gsl-lite fmt::fmt-header-only Torch)
    target_compile_definitions(nqs_gpu PUBLIC TCM_USE_CUDA)
else()
    add_library(nqs_gpu INTERFACE)
endif()
