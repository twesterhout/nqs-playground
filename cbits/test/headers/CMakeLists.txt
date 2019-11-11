
function(add_header_test my_name)
    add_library(${my_name}-header OBJECT ${my_name}-header.cpp)
    target_link_libraries(${my_name}-header PRIVATE nqs_cbits_Common)
endfunction()

add_header_test(common)
add_header_test(config)
add_header_test(errors)
add_header_test(spin_basis)
target_link_libraries(spin_basis-header PRIVATE pybind11::pybind11)

add_header_test(quantum_state)
target_link_libraries(quantum_state-header PRIVATE pybind11::pybind11)

add_header_test(polynomial)
target_link_libraries(polynomial-header PRIVATE pybind11::pybind11)

if(FALSE)
    add_library(NQS_common INTERFACE)
    target_compile_options(NQS_common INTERFACE ${TCM_WARNING_FLAGS})
    target_compile_options(NQS_common INTERFACE "-fvisibility=hidden")
    target_compile_options(NQS_common INTERFACE "-msse" "-msse2" "-msse3" "-msse4.1" "-msse4.2" "-mavx" "-mtune=native")
    target_compile_definitions(NQS_common INTERFACE
        TORCH_API_INCLUDE_EXTENSION_H=1
        TORCH_EXTENSION_NAME="_C_nqs"
        _GLIBCXX_USE_CXX11_ABI=0
    )

    add_executable(config-header config-header.cpp)
    target_link_libraries(config-header NQS_common)
    target_link_libraries(config-header Boost)

    add_executable(errors-header ../../errors.cpp errors-header.cpp)
    target_link_libraries(errors-header NQS_common)
    target_link_libraries(errors-header fmt::fmt Boost Torch)

    add_executable(parallel-header parallel-header.cpp)
    target_link_libraries(parallel-header NQS_common)
    target_link_libraries(parallel-header OpenMP::OpenMP_CXX fmt::fmt Boost Torch)

    add_executable(random-header random-header.cpp)
    target_link_libraries(random-header NQS_common)

    add_library(spin-header ../../spin.cpp spin-header.cpp)
    target_link_libraries(spin-header NQS_common)
    target_link_libraries(spin-header gsl-lite fmt::fmt Boost Torch pybind11::pybind11)

    add_library(polynomial-header ../../polynomial.cpp polynomial-header.cpp)
    target_link_libraries(polynomial-header NQS_common)
    target_link_libraries(polynomial-header OpenMP::OpenMP_CXX gsl-lite SG14 fmt::fmt Boost Torch pybind11::pybind11 flat_hash_map)

    add_library(nn-header ../../nn.cpp nn-header.cpp)
    target_link_libraries(nn-header NQS_common)
    target_link_libraries(nn-header OpenMP::OpenMP_CXX fmt::fmt Boost Torch Sleef pybind11::pybind11)
    target_include_directories(nn-header SYSTEM PUBLIC /vol/tcm01/westerhout_tom/conda/pkgs/mkl-include-2019.3-199/include)

    add_library(monte_carlo-header ../../monte_carlo.cpp monte_carlo-header.cpp)
    target_link_libraries(monte_carlo-header NQS_common)
    target_link_libraries(monte_carlo-header OpenMP::OpenMP_CXX gsl-lite SG14
        fmt::fmt Boost Torch Sleef pybind11::pybind11 ska_sort flat_hash_map)
    target_include_directories(monte_carlo-header SYSTEM PUBLIC /vol/tcm01/westerhout_tom/conda/pkgs/mkl-include-2019.3-199/include)

    add_library(data-header ../../data.cpp data-header.cpp)
    target_link_libraries(data-header NQS_common)
    target_link_libraries(data-header OpenMP::OpenMP_CXX gsl-lite SG14
        fmt::fmt Boost Torch Sleef pybind11::pybind11 ska_sort flat_hash_map)
    target_include_directories(data-header SYSTEM PUBLIC /vol/tcm01/westerhout_tom/conda/pkgs/mkl-include-2019.3-199/include)
endif()
