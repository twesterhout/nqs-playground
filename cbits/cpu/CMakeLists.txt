# Library of CPU specific kernels

foreach(my_arch sse2 avx avx2)
    set(my_target "nqs_cpu_kernels_${my_arch}")
    add_library(${my_target} OBJECT
        kernels.hpp
        kernels.cpp
    )
    nqs_cbits_add_low_level_flags(${my_target})
    target_compile_options(${my_target} PRIVATE ${TCM_WARNING_FLAGS})
    target_compile_definitions(${my_target} PRIVATE TCM_COMPILE_${my_arch})
    target_link_libraries(${my_target} PUBLIC nqs_cbits_Common)
    target_link_libraries(${my_target} PRIVATE vectorclass)
endforeach()
target_compile_options(nqs_cpu_kernels_sse2 PRIVATE -m64 -msse2)
target_compile_definitions(nqs_cpu_kernels_sse2 PRIVATE TCM_ADD_DISPATCH_CODE)
target_compile_options(nqs_cpu_kernels_avx PRIVATE -m64 -mavx)
target_compile_options(nqs_cpu_kernels_avx2 PRIVATE -m64 -mavx2 -mfma)
