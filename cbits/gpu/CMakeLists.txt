# Determine if we have nvcc
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_FLAGS "")
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
    target_compile_options(nqs_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin -compress-all
                                   -gencode arch=compute_35,code=sm_35>)
    target_link_libraries(nqs_gpu PUBLIC nqs_cbits_Common)
    target_compile_definitions(nqs_gpu PUBLIC TCM_USE_CUDA)
else()
    add_library(nqs_gpu INTERFACE)
endif()
