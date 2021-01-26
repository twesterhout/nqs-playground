# Library of CPU specific kernels

foreach(my_arch sse4 avx avx2)
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
target_compile_options(nqs_cpu_kernels_sse4 PRIVATE -msse4 -msse4.1 -msse4.2 -mpopcnt)
target_compile_definitions(nqs_cpu_kernels_sse4 PRIVATE TCM_ADD_DISPATCH_CODE)
target_compile_options(nqs_cpu_kernels_avx PRIVATE -mavx)
target_compile_options(nqs_cpu_kernels_avx2 PRIVATE -mavx2 -mbmi2 -mfma)
