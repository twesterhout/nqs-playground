cmake_minimum_required(VERSION 3.9)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" TCM_BUILD_TYPE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

project(nqs_cbits)
include(ExternalProject)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(TCM_WARNING_FLAGS -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-unused
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

# {fmt} library
execute_process(COMMAND git submodule update --init -- external/fmt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
option(FMT_PEDANTIC ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/fmt)
target_compile_definitions(fmt PUBLIC
    _GLIBCXX_USE_CXX11_ABI=0)

# gsl-lite library
execute_process(COMMAND git submodule update --init -- external/gsl-lite
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_library(gsl-lite INTERFACE)
target_include_directories(gsl-lite
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/gsl-lite/include)
if("${TCM_BUILD_TYPE}" STREQUAL "DEBUG")
    target_compile_definitions(gsl-lite INTERFACE
        gsl_CONFIG_CONTRACT_LEVEL_ON=1
        gsl_CONFIG_CONTRACT_VIOLATION_THROWS=1)
else()
    target_compile_definitions(gsl-lite INTERFACE
        gsl_CONFIG_CONTRACT_LEVEL_OFF=1)
    message(STATUS "Disabling GSL contracts")
endif()

# flat_hash_map library
execute_process(COMMAND git submodule update --init -- external/flat_hash_map
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_library(flat_hash_map INTERFACE)
target_include_directories(flat_hash_map
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external)

# cpuinfo library
execute_process(COMMAND git submodule update --init -- external/cpuinfo
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CPUINFO_LIBRARY_TYPE "static")
option(CPUINFO_BUILD_TOOLS OFF)
option(CPUINFO_BUILD_UNIT_TESTS OFF)
option(CPUINFO_BUILD_MOCK_TESTS OFF)
option(CPUINFO_BUILD_BENCHMARKS OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/cpuinfo)

# Boost (some header-only parts)
execute_process(COMMAND git submodule update --init -- external/boost
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
execute_process(COMMAND git submodule update --init --
    libs/config libs/core libs/detail libs/pool libs/utility
    libs/range libs/serialization libs/static_assert libs/type_traits
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/boost)
add_library(Boost INTERFACE)
target_include_directories(Boost SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/config/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/detail/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/pool/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/utility/include)

# Torch
add_library(Torch INTERFACE)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" "-c"
    "from torch.utils.cpp_extension import include_paths;
print(';'.join(include_paths()), end='')"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUE
    ERROR_VARIABLE _PYTHON_ERROR_VALUE)
if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR
        "Could not find PyTorch include directories:\n${_PYTHON_ERROR_VALUE}")
    return()
endif()
target_include_directories(Torch SYSTEM INTERFACE ${_PYTHON_VALUE})
execute_process(COMMAND "${Python3_EXECUTABLE}" "-c"
    "from torch.utils.cpp_extension import library_paths;
print(';'.join(library_paths()), end='')"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUE
    ERROR_VARIABLE _PYTHON_ERROR_VALUE)
if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR
        "Could not find PyTorch library paths:\n${_PYTHON_ERROR_VALUE}")
    return()
endif()
target_link_libraries(Torch INTERFACE ${_PYTHON_VALUE})

# Pybind11
execute_process(COMMAND git submodule update --init -- external/pybind11
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(PYBIND11_CPP_STANDARD "-std=c++14")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)

# add_library(nqs SHARED cbits/nqs.cpp)
# if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     target_compile_options(nqs PRIVATE -fsized-deallocation)
# endif()
# target_compile_options(nqs PRIVATE ${TCM_WARNING_FLAGS})
# target_compile_options(nqs PRIVATE "-march=native")
# target_include_directories(nqs SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
# message(STATUS "Python3 includes: ${Python3_INCLUDE_DIRS}")
# target_link_libraries(nqs PRIVATE fmt gsl-lite flat_hash_map cpuinfo Boost Torch)

pybind11_add_module(_C_nqs MODULE SYSTEM cbits/nqs.cpp)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(_C_nqs PRIVATE -fsized-deallocation)
endif()
target_compile_options(_C_nqs PRIVATE ${TCM_WARNING_FLAGS})
target_compile_options(_C_nqs PRIVATE "-march=native")
target_compile_options(_C_nqs PRIVATE "-fopenmp")
target_compile_definitions(_C_nqs PUBLIC
    TORCH_API_INCLUDE_EXTENSION_H=1
    TORCH_EXTENSION_NAME="_C_nqs"
    _GLIBCXX_USE_CXX11_ABI=0
    _GLIBCXX_PARALLEL=1)
target_link_libraries(_C_nqs PUBLIC fmt::fmt gsl-lite flat_hash_map cpuinfo Boost Torch)

if (FALSE)
    set(PYBIND11_CPP_STANDARD "-std=c++14")
    add_subdirectory(external/pybind11)

    pybind11_add_module(_C_nqs MODULE SYSTEM cbits/nqs.cpp)
endif()
