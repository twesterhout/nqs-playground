cmake_minimum_required(VERSION 3.13)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" TCM_BUILD_TYPE)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0104 NEW)
project(nqs_cbits
    DESCRIPTION "_C Python extension for nqs_playground"
    LANGUAGES C CXX)

option(NQS_PLAYGROUND_USE_CUDA "Try to compile with GPU support" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

include(CheckCXXCompilerFlag)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(TCM_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-compat -Wno-weak-vtables -Wno-unused
        -Wno-exit-time-destructors -Wno-documentation-unknown-command
        -Wno-float-equal -Wno-global-constructors -Wno-padded
        -Wno-undefined-func-template)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(TCM_WARNING_FLAGS -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
        -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=2 -Wswitch-default -Wundef -Wno-unused)
endif()

###############################################################################
# Dependencies
###############################################################################

# Torch
if(FALSE)
    # This requires at least CMake 3.12
    find_package(Python3 COMPONENTS Development REQUIRED)
    if (NOT DEFINED Python3_EXECUTABLE)
        message(STATUS "Setting Python3_EXECUTABLE manually...")
        set(Python3_EXECUTABLE "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    endif()

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

    add_library(Torch INTERFACE)
    run_python_command("from torch.utils.cpp_extension import include_paths; \
print(';'.join(include_paths(cuda=True)))" _TEMP)
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

    run_python_command("import torch; import re; \
a, b = map(int, torch.__version__.split('.')[:-1]); \
print(1000 * (1000 * a + b))" _TEMP)
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
    # add_library(torch ALIAS Torch)

    message(STATUS "PATH: $ENV{CONDA_PREFIX}/lib/python3.8/lib/site-packages/torch/lib")
    find_library(libc10 c10
        HINTS $ENV{CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib
        REQUIRED
    )
    find_library(libtorch torch
        HINTS $ENV{CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib
        REQUIRED
    )
    find_library(libtorch_python torch_python
        HINTS $ENV{CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib
        REQUIRED
    )
    message(STATUS "libc10: ${libc10}")

    target_link_libraries(Torch INTERFACE ${libtorch_python} ${libtorch} ${libc10})
else()
    # This doesn't work properly, because PyTorch is hardcoding some paths in
    # Caffe2Config.cmake
    find_package(Torch REQUIRED)
    message(STATUS "TORCH_LIBRARY: ${TORCH_LIBRARY}")
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
    message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
endif()

if(TRUE)
    find_package(PkgConfig REQUIRED) 
    pkg_check_modules(lattice_symmetries REQUIRED IMPORTED_TARGET lattice_symmetries)
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
        PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

# {fmt} library
# if(NOT TARGET fmt::fmt)
    execute_process(COMMAND git submodule update --init -- external/fmt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    option(FMT_PEDANTIC ON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/fmt)
    nqs_cbits_add_low_level_flags(fmt)
    # Mark {fmt} include directories as system. Otherwise, we'd get a whole lot
    # of warnings...
    get_property(v TARGET fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET fmt PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(fmt SYSTEM INTERFACE ${v})
    get_property(v TARGET fmt-header-only PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET fmt-header-only PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(fmt-header-only SYSTEM BEFORE INTERFACE ${v})
# endif()

# gsl-lite library
if(NOT TARGET gsl::gsl-lite)
    execute_process(COMMAND git submodule update --init -- external/gsl-lite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(gsl-lite INTERFACE)
    target_include_directories(gsl-lite
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/gsl-lite/include)
    if("${TCM_BUILD_TYPE}" STREQUAL "DEBUG")
        target_compile_definitions(gsl-lite INTERFACE
            gsl_CONFIG_CONTRACT_CHECKING_ON=1
            gsl_CONFIG_CONTRACT_VIOLATION_THROWS=1)
        message(STATUS "[nqs_cbits] GSL will throw on contract violation...")
    else()
        target_compile_definitions(gsl-lite INTERFACE
            gsl_CONFIG_CONTRACT_CHECKING_OFF=1)
        message(STATUS "[nqs_cbits] disabling GSL contracts...")
    endif()
    add_library(gsl::gsl-lite ALIAS gsl-lite)
endif()

# vectorclass library
if(NOT TARGET vectorclass)
    execute_process(COMMAND git submodule update --init -- external/vectorclass/version2
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # execute_process(COMMAND git submodule update --init -- external/vectorclass/add-on
    #     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(vectorclass INTERFACE)
    target_include_directories(vectorclass
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external)
    target_compile_definitions(vectorclass INTERFACE VCL_NAMESPACE=vcl)
endif()

# SG14 library
if(NOT TARGET SG14)
    execute_process(COMMAND git submodule update --init -- external/SG14
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(SG14 INTERFACE)
    target_include_directories(SG14
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/SG14)
endif()

# Pybind11
if(TRUE)
    execute_process(COMMAND git submodule update --init -- external/pybind11
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(PYBIND11_CPP_STANDARD "-std=c++17")
    set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
endif()

# OpenMP
find_package(OpenMP REQUIRED)


add_library(nqs_cbits_Common INTERFACE)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(nqs_cbits_Common INTERFACE -fsized-deallocation)
endif()
target_link_libraries(nqs_cbits_Common INTERFACE
    INTERFACE torch
              fmt
              gsl::gsl-lite
              SG14
              PkgConfig::lattice_symmetries
              OpenMP::OpenMP_CXX
)


###############################################################################
# Library of CUDA specific kernels
###############################################################################
add_subdirectory(cbits/gpu)


###############################################################################
# Library of CPU specific kernels
###############################################################################
add_subdirectory(cbits/cpu)


###############################################################################
# General Python-independent code
###############################################################################
add_subdirectory(cbits/common)


###############################################################################
# Python bindings
###############################################################################
add_subdirectory(cbits/python)


###############################################################################
# Installing
###############################################################################
set_property(TARGET _C
    PROPERTY INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/nqs_playground")
set_property(TARGET _C PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property(TARGET nqs PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS nqs _C
        LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/nqs_playground)
