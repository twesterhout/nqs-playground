cmake_minimum_required(VERSION 3.13)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" TCM_BUILD_TYPE)

cmake_policy(SET CMP0048 NEW)
project(nqs_cbits
    VERSION 0.0.1
    DESCRIPTION "_C Python extension for nqs-playground"
    LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# C++17 without extensions
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

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

include(CTest)
include(CheckCXXCompilerFlag)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(TCM_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-compat -Wno-weak-vtables -Wno-unused
        -Wno-exit-time-destructors -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-global-constructors -Wno-padded
        )
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(TCM_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused)
endif()

if(UNIX AND CMAKE_GENERATOR STREQUAL "Ninja")
    # if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    #     list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
    #     list(APPEND CMAKE_C_FLAGS "-fcolor-diagnostics")
    # elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    #     list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
    #     list(APPEND CMAKE_C_FLAGS "-fdiagnostics-color=always")
    # endif()
endif()


# list(APPEND CMAKE_PREFIX_PATH
#     "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/torch")
# message(STATUS "${CMAKE_PREFIX_PATH}")


# add_library(Torch INTERFACE)
# set(TORCH_PREFIX "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/torch")
# target_include_directories(Torch SYSTEM INTERFACE ${TORCH_PREFIX})
# target_compile_options()

# Torch
if(TRUE)
    # This requires at least CMake 3.12
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Runs python command _COMMAND and stores its output into _RESULT
    function(run_python_command _COMMAND _RESULT)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "${_COMMAND}"
            RESULT_VARIABLE _SUCCESS
            OUTPUT_VARIABLE _VALUE
            ERROR_VARIABLE _ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _SUCCESS MATCHES 0)
            message(FATAL_ERROR
                "Could not find PyTorch include directories:\n${_ERROR_VALUE}")
            return()
        endif()
        set(${_RESULT} "${_VALUE}" PARENT_SCOPE)
    endfunction()

    # Stores the last line of _STRING into _RESULT
    function(get_last_line _STRING _RESULT)
        string(REGEX REPLACE "^(.*)\n" "" _VALUE "${_STRING}")
        set(${_RESULT} "${_VALUE}" PARENT_SCOPE)
    endfunction()

    # We don't use CUDA in our C++ code, so everything's simple
    add_library(Torch INTERFACE)
    run_python_command("from torch.utils.cpp_extension import include_paths; \
print(';'.join(include_paths(cuda=False)))" _TEMP)
    # We need get_last_line here, because PyTorch prints warnings when CUDA_HOME
    # is set, but it can't find CUDA runtime. This happens when we compile code
    # on Cartesius on nodes which don't have GPUs but still load CUDA module.
    # The warning doesn't apply to use since we don't use CUDA here.
    get_last_line("${_TEMP}" _TORCH_INCLUDE_DIRS)
    # This part is important!!
    run_python_command("import torch; \
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" _TEMP)
    # Same here
    get_last_line("${_TEMP}" TCM_USE_CXX11_ABI)

    run_python_command("import torch; \
a, b, c, = map(int, torch.__version__.split('.')); \
print(1000 * (1000 * a + b) + c)" _TEMP)
    get_last_line("${_TEMP}" _TORCH_VERSION)


    message(STATUS "Torch version: ${_TORCH_VERSION}")
    message(STATUS "Torch include directories: ${_TORCH_INCLUDE_DIRS}")
    message(STATUS "[nqs_cbits] using C++11 abi: ${TCM_USE_CXX11_ABI}")
    target_include_directories(Torch SYSTEM INTERFACE ${_TORCH_INCLUDE_DIRS})
    target_compile_definitions(Torch INTERFACE
        TCM_TORCH_VERSION=${_TORCH_VERSION}
        _GLIBCXX_USE_CXX11_ABI=${TCM_USE_CXX11_ABI})
    # Because PyTorch adds torch/include/THC, but it's not available on cpuonly
    # PyTorch installations
    # target_compile_options(Torch INTERFACE -Wno-missing-include-dirs)
    add_library(torch ALIAS Torch)

else()
    # This doesn't work properly, because PyTorch is hardcoding some paths in
    # Caffe2Config.cmake
    find_package(Torch REQUIRED)
endif()

function(nqs_cbits_add_low_level_flags MY_TARGET)
    if (TCM_USE_CXX11_ABI)
        target_compile_definitions(${MY_TARGET} PUBLIC
            _GLIBCXX_USE_CXX11_ABI=${TCM_USE_CXX11_ABI})
    endif()
    if (TORCH_CXX_FLAGS)
        target_compile_options(${MY_TARGET} PUBLIC ${TORCH_CXX_FLAGS})
    endif()
    set_property(TARGET ${MY_TARGET}
        PROPERTY CXX_VISIBILITY_PRESET hidden)
    set_property(TARGET ${MY_TARGET}
        PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

add_library(nqs_cbits_Common INTERFACE)
# TODO: The following is rather unportable and should be fixed...
# target_compile_options(nqs_cbits_Common
#     INTERFACE -mmmx
#               -msse -msse2 -msse3 -msse4.1 -msse4.2
#               -mavx -march=ivybridge -mtune=native)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(nqs_cbits_Common INTERFACE -fsized-deallocation)
endif()
# target_compile_options(nqs_cbits_Common INTERFACE ${TCM_WARNING_FLAGS})
# target_include_directories(nqs_cbits_Common SYSTEM INTERFACE $ENV{CONDA_PREFIX}/include)

# {fmt} library
if(NOT TARGET fmt::fmt)
    execute_process(COMMAND git submodule update --init -- external/fmt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    option(FMT_PEDANTIC ON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/fmt)
    nqs_cbits_add_low_level_flags(fmt)
    # Mark {fmt} include directories as system. Otherwise, we'd get a whole lot
    # of warnings...
    get_property(v TARGET fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(fmt SYSTEM BEFORE INTERFACE ${v})
endif()

# gsl-lite library
if(NOT TARGET gsl::gsl-lite)
    execute_process(COMMAND git submodule update --init -- external/gsl-lite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(gsl-lite INTERFACE)
    target_include_directories(gsl-lite
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/gsl-lite/include)
    if("${TCM_BUILD_TYPE}" STREQUAL "DEBUG")
        target_compile_definitions(gsl-lite INTERFACE
            gsl_CONFIG_CONTRACT_LEVEL_ON=1
            gsl_CONFIG_CONTRACT_VIOLATION_THROWS=1)
        message(STATUS "[nqs_cbits] GSL will throw on contract violation...")
    else()
        target_compile_definitions(gsl-lite INTERFACE
            gsl_CONFIG_CONTRACT_LEVEL_OFF=1)
        message(STATUS "[nqs_cbits] disabling GSL contracts...")
    endif()
    add_library(gsl::gsl-lite ALIAS gsl-lite)
endif()

# expected library
if(NOT TARGET tl::expected)
    execute_process(COMMAND git submodule update --init -- external/expected
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    option(EXPECTED_ENABLE_TESTS OFF)
    option(EXPECTED_ENABLE_DOCS OFF)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/expected)
endif()

# flat_hash_map library
if(NOT TARGET flat_hash_map)
    # Can't use submodules here anymore, thanks to PyTorch deciding to include
    # flat_hash_map in their codebase...
    # execute_process(COMMAND git submodule update --init -- external/flat_hash_map
    #     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(flat_hash_map INTERFACE)
    target_include_directories(flat_hash_map
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/flat_hash_map)
endif()

# ska_sort library
# if(NOT TARGET ska_sort)
#     execute_process(COMMAND git submodule update --init -- external/ska_sort
#         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#     add_library(ska_sort INTERFACE)
#     target_include_directories(ska_sort
#         SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/ska_sort)
# endif()

if(NOT TARGET vectorclass)
    execute_process(COMMAND git submodule update --init -- external/vectorclass/version2
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    execute_process(COMMAND git submodule update --init -- external/vectorclass/add-on
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(vectorclass INTERFACE)
    target_include_directories(vectorclass
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external)
    target_compile_definitions(vectorclass INTERFACE VCL_NAMESPACE=vcl)
endif()

# cpuinfo library
# execute_process(COMMAND git submodule update --init -- external/cpuinfo
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# set(CPUINFO_LIBRARY_TYPE "static")
# option(CPUINFO_BUILD_TOOLS OFF)
# option(CPUINFO_BUILD_UNIT_TESTS OFF)
# option(CPUINFO_BUILD_MOCK_TESTS OFF)
# option(CPUINFO_BUILD_BENCHMARKS OFF)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/cpuinfo EXCLUDE_FROM_ALL)
# target_compile_definitions(cpuinfo PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
# target_compile_options(cpuinfo PUBLIC "-fPIC")

# SG14 library
if(NOT TARGET SG14)
    execute_process(COMMAND git submodule update --init -- external/SG14
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(SG14 INTERFACE)
    target_include_directories(SG14
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/SG14)
endif()

# ThreadPool
if(NOT TARGET ThreadPool)
    find_package(Threads REQUIRED)
    execute_process(COMMAND git submodule update --init -- external/ThreadPool
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(ThreadPool INTERFACE)
    target_include_directories(ThreadPool
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external)
    target_link_libraries(ThreadPool INTERFACE Threads::Threads)
endif()

# Boost libraries (some header-only parts)
if(NOT TARGET Boost::boost)
    execute_process(COMMAND git submodule update --init -- external/boost
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    execute_process(COMMAND git submodule update --init -- libs/align
                                                           libs/array
                                                           libs/assert
                                                           libs/concept_check
                                                           libs/config
                                                           libs/container
                                                           libs/core
                                                           libs/detail
                                                           libs/integer
                                                           libs/intrusive
                                                           libs/iterator
                                                           libs/lexical_cast
                                                           libs/math
                                                           libs/move
                                                           libs/mpl
                                                           libs/numeric/conversion
                                                           libs/pool
                                                           libs/predef
                                                           libs/preprocessor
                                                           libs/range
                                                           libs/smart_ptr
                                                           libs/static_assert
                                                           libs/throw_exception
                                                           libs/type_traits
                                                           libs/utility 
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/boost)
    add_library(Boost INTERFACE)
    target_include_directories(Boost SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/align/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/array/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/assert/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/concept_check/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/config/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/container/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/detail/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/integer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/intrusive/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/iterator/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/lexical_cast/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/math/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/move/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/mpl/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/numeric/conversion/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/pool/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/predef/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/preprocessor/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/range/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/smart_ptr/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/static_assert/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/throw_exception/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/type_traits/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/utility/include)
    add_library(Boost::boost ALIAS Boost)
endif()


# Pybind11
if(TRUE)
    execute_process(COMMAND git submodule update --init -- external/pybind11
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(PYBIND11_CPP_STANDARD "-std=c++17")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# Taskflow
if(NOT TARGET Taskflow)
    find_package(Threads REQUIRED)
    execute_process(COMMAND git submodule update --init -- external/cpp-taskflow
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(Taskflow INTERFACE)
    target_include_directories(Taskflow
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-taskflow)
    target_link_libraries(Taskflow INTERFACE Threads::Threads)
endif()

# Sleef
if(FALSE)
ExternalProject_Add(Sleef_build
  GIT_REPOSITORY https://github.com/shibatch/sleef
  BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/external/lib/libsleef.a
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_DFT=OFF
    -DBUILD_QUAD=OFF
    -DBUILD_GNUABI_LIBS=OFF
    -DBUILD_TESTS=OFF
)
add_library(Sleef INTERFACE)
add_dependencies(Sleef Sleef_build)
target_include_directories(Sleef SYSTEM INTERFACE ${CMAKE_BINARY_DIR}/external/include)
target_link_libraries(Sleef INTERFACE ${CMAKE_BINARY_DIR}/external/lib/libsleef.a)
endif()

target_link_libraries(nqs_cbits_Common INTERFACE
    INTERFACE torch
              fmt::fmt
              gsl::gsl-lite
              flat_hash_map
              # ska_sort
              Boost::boost
              SG14
              ThreadPool
              Taskflow
              OpenMP::OpenMP_CXX
)

set(nqs_cbits_HEADERS)

###############################################################################
# Library of CUDA specific kernels
###############################################################################
if(CMAKE_CUDA_COMPILER)
    add_library(nqs_gpu STATIC
        cbits/gpu/unpack.hpp
        cbits/gpu/unpack.cu)
    target_compile_definitions(nqs_gpu PUBLIC
        _GLIBCXX_USE_CXX11_ABI=${TCM_USE_CXX11_ABI})
    # TODO: Fix warnings
    target_compile_options(nqs_gpu PRIVATE
    	--compiler-options "-W -Wall -Wextra")
    set_property(TARGET nqs_gpu
        PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(nqs_gpu PUBLIC Boost::boost fmt::fmt torch)
endif()


###############################################################################
# Library of CPU specific kernels
###############################################################################
add_library(nqs_cpu STATIC
    cbits/cpu/unpack.hpp
    cbits/cpu/unpack.cpp)
target_compile_definitions(nqs_cpu PUBLIC
    _GLIBCXX_USE_CXX11_ABI=${TCM_USE_CXX11_ABI})
target_compile_options(nqs_cpu PRIVATE ${TCM_WARNING_FLAGS})
set_property(TARGET nqs_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)
# Efficient AVX implementation
add_library(nqs_cpu_avx STATIC
    cbits/cpu/unpack.hpp
    cbits/cpu/unpack_avx.cpp)
target_compile_definitions(nqs_cpu_avx PUBLIC
    _GLIBCXX_USE_CXX11_ABI=${TCM_USE_CXX11_ABI})
target_compile_options(nqs_cpu_avx PRIVATE ${TCM_WARNING_FLAGS})
set_property(TARGET nqs_cpu_avx PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nqs_cpu_avx PUBLIC torch fmt::fmt Boost::boost vectorclass)
target_compile_options(nqs_cpu_avx PRIVATE
    -mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mtune=haswell)
# And use it
target_link_libraries(nqs_cpu PRIVATE nqs_cpu_avx)


###############################################################################
# General Python independent code
###############################################################################
add_library(nqs SHARED
    cbits/bits512.cpp
    cbits/errors.cpp
    cbits/metropolis.cpp
    cbits/jacobian.cpp
    cbits/heisenberg.cpp
    cbits/forward_propagator.cpp
    cbits/polynomial_state.cpp
    cbits/random.cpp
    cbits/unpack.cpp
    # cbits/operators.cpp
    cbits/symmetry.cpp
    cbits/spin_basis.cpp
    cbits/parallel.cpp
    cbits/pool.cpp
    cbits/tensor_info.cpp
    cbits/trim.cpp
)
nqs_cbits_add_low_level_flags(nqs)
target_compile_definitions(nqs PUBLIC
    TORCH_API_INCLUDE_EXTENSION_H=1
    TORCH_EXTENSION_NAME="_C")
target_compile_options(nqs PRIVATE ${TCM_WARNING_FLAGS})
target_link_libraries(nqs PUBLIC nqs_cbits_Common nqs_cpu)
if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(nqs PUBLIC TCM_USE_CUDA)
    target_link_libraries(nqs PUBLIC nqs_gpu)
endif()


pybind11_add_module(_C MODULE SYSTEM NO_EXTRAS
    cbits/python/bind_symmetry.cpp
    cbits/python/bind_heisenberg.cpp
    cbits/python/bind_metropolis.cpp
    cbits/python/bind_spin_basis.cpp
    cbits/python/bind_polynomial.cpp
    cbits/python/bind_polynomial_state.cpp
    cbits/python/bind_jacobian.cpp
    cbits/python/bind_cuda.cpp
    cbits/python/init.cpp
)
# pybind11_add_module(_C_nqs MODULE SYSTEM NO_EXTRAS
#     cbits/nqs.cpp
#     # cbits/data.cpp
#     cbits/errors.cpp
#     cbits/metropolis.cpp
#     cbits/heisenberg.cpp
#     # cbits/monte_carlo_v2.cpp
#     # cbits/nn.cpp
#     # cbits/polynomial.cpp
#     # cbits/polynomial_state.cpp
#     cbits/random.cpp
#     cbits/operators.cpp
#     cbits/symmetry.cpp
#     cbits/spin_basis.cpp
#     cbits/pool.cpp
#     cbits/python/bind_symmetry.cpp
#     cbits/trim.cpp
# )
nqs_cbits_add_low_level_flags(_C)
target_compile_definitions(_C PUBLIC
    TORCH_API_INCLUDE_EXTENSION_H=1
    TORCH_EXTENSION_NAME="_C")
# We need to link against libtorch_python.so as well since _C is a Python
# extension
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# target_link_libraries(_C PUBLIC nqs nqs_cbits_Common ${TORCH_PYTHON_LIBRARY})
target_link_libraries(_C PUBLIC nqs)

add_subdirectory(cbits/test/headers)

set_property(TARGET _C
    PROPERTY INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/nqs_playground")
set_property(TARGET _C PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property(TARGET nqs PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS nqs _C
        LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/nqs_playground)
