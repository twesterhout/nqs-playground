# General Python-independent code
add_library(nqs SHARED
    bits512.hpp
    bits512.cpp
    errors.hpp
    errors.cpp
    metropolis.hpp
    metropolis.cpp
    zanella.hpp
    zanella.cpp
    accumulator.hpp
    accumulator.cpp
    polynomial.hpp
    polynomial.cpp
    unpack.cpp
    parallel.hpp
    parallel.cpp
    random.hpp
    random.cpp
    wrappers.hpp
    wrappers.cpp
)
nqs_cbits_add_low_level_flags(nqs)
target_compile_options(nqs PRIVATE ${TCM_WARNING_FLAGS})
target_link_libraries(nqs PUBLIC nqs_cbits_Common
                                 nqs_gpu
                                 nqs_cpu_kernels_avx2
                                 nqs_cpu_kernels_avx
                                 nqs_cpu_kernels_sse4)
