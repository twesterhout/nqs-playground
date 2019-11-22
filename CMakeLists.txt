cmake_minimum_required(VERSION 3.13)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" TCM_BUILD_TYPE)

project(nqs_cbits)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CTest)
include(CheckCXXCompilerFlag)
include(ExternalProject)

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

find_package(Python COMPONENTS Interpreter REQUIRED)

list(APPEND CMAKE_PREFIX_PATH
    "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/torch")
message(STATUS "${CMAKE_PREFIX_PATH}")


# add_library(Torch INTERFACE)
# set(TORCH_PREFIX "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/torch")
# target_include_directories(Torch SYSTEM INTERFACE ${TORCH_PREFIX})
# target_compile_options()

# Torch
if(TRUE)
    add_library(Torch INTERFACE)
    execute_process(COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/get_torch_build_info.sh"
            "${Python_EXECUTABLE}" "include"
        RESULT_VARIABLE _SUCCESS
        OUTPUT_VARIABLE _VALUE
        ERROR_VARIABLE _ERROR_VALUE
	OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _SUCCESS MATCHES 0)
        message(FATAL_ERROR
            "Could not find PyTorch include directories:\n${_ERROR_VALUE}")
        return()
    endif()
    target_include_directories(Torch SYSTEM INTERFACE ${_VALUE})
    execute_process(COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/get_torch_build_info.sh"
            "${Python_EXECUTABLE}" "lib"
        RESULT_VARIABLE _SUCCESS
        OUTPUT_VARIABLE _VALUE
        ERROR_VARIABLE _ERROR_VALUE
	OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _SUCCESS MATCHES 0)
        message(FATAL_ERROR
            "Could not find PyTorch include directories:\n${_ERROR_VALUE}")
        return()
    endif()
    target_link_libraries(Torch INTERFACE ${_VALUE})
    execute_process(COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/get_torch_build_info.sh"
            "${Python_EXECUTABLE}" "abi"
        RESULT_VARIABLE _SUCCESS
        OUTPUT_VARIABLE _VALUE
        ERROR_VARIABLE _ERROR_VALUE
	OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _SUCCESS MATCHES 0)
        message(FATAL_ERROR
            "Could not find PyTorch include directories:\n${_ERROR_VALUE}")
        return()
    endif()
    target_compile_definitions(Torch INTERFACE _GLIBCXX_USE_CXX11_ABI=${_VALUE})
    add_library(torch ALIAS Torch)
endif()

function(nqs_cbits_add_low_level_flags MY_TARGET)
    # Since Torch might've been compiled with old ABI
    foreach(prop # COMPILE_DEFINITIONS
    		 INTERFACE_COMPILE_DEFINITIONS)
        get_property(s TARGET torch PROPERTY ${prop} SET)
        if(s)
            get_property(v TARGET torch PROPERTY ${prop})
            target_compile_definitions(${MY_TARGET} PUBLIC ${v})
        endif()
    endforeach()
    foreach(prop # COMPILE_OPTIONS
                 INTERFACE_COMPILE_OPTIONS)
        get_property(s TARGET torch PROPERTY ${prop} SET)
        if(s)
            get_property(v TARGET torch PROPERTY ${prop})
            target_compile_options(${MY_TARGET} PUBLIC ${v})
        endif()
    endforeach()
    set_property(TARGET ${MY_TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

add_library(nqs_cbits_Common INTERFACE)
# TODO: The following is rather unportable and should be fixed...
target_compile_options(nqs_cbits_Common
    INTERFACE -mmmx
              -msse -msse2 -msse3 -msse4.1 -msse4.2
              -mavx -march=ivybridge -mtune=native)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(nqs_cbits_Common INTERFACE -fsized-deallocation)
endif()
target_compile_options(nqs_cbits_Common INTERFACE ${TCM_WARNING_FLAGS})
target_include_directories(nqs_cbits_Common SYSTEM INTERFACE $ENV{CONDA_PREFIX}/include)

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
    set_property(TARGET fmt PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${v})
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
if(NOT TARGET ska_sort)
    # execute_process(COMMAND git submodule update --init -- external/ska_sort/ska_sort
    #     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(ska_sort INTERFACE)
    target_include_directories(ska_sort
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/ska_sort)
endif()

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

# Boost libraries (some header-only parts)
if(NOT TARGET Boost::boost)
    execute_process(COMMAND git submodule update --init -- external/boost
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    execute_process(COMMAND git submodule update --init -- libs/align
                                                           libs/assert
                                                           libs/config
                                                           libs/container
                                                           libs/core
                                                           libs/detail
                                                           libs/integer
                                                           libs/move
                                                           libs/pool
                                                           libs/static_assert
                                                           libs/throw_exception
                                                           libs/utility 
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/boost)
    add_library(Boost INTERFACE)
    target_include_directories(Boost SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/align/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/assert/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/config/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/container/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/detail/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/integer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/move/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/pool/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/static_assert/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/throw_exception/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost/libs/utility/include)
    add_library(Boost::boost ALIAS Boost)
endif()


# Pybind11
if(TRUE)
    execute_process(COMMAND git submodule update --init -- external/pybind11
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(PYBIND11_CPP_STANDARD "-std=c++14")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# Taskflow
if(NOT TARGET Taskflow)
    execute_process(COMMAND git submodule update --init -- external/cpp-taskflow
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_library(Taskflow INTERFACE)
    target_include_directories(Taskflow
        SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-taskflow)
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
              ska_sort
              Boost::boost
              SG14
              vectorclass
              Taskflow
              OpenMP::OpenMP_CXX
)

set(nqs_cbits_HEADERS)

pybind11_add_module(_C_nqs MODULE SYSTEM NO_EXTRAS
    cbits/nqs.cpp
    # cbits/data.cpp
    cbits/errors.cpp
    # cbits/monte_carlo.cpp
    cbits/monte_carlo_v2.cpp
    # cbits/nn.cpp
    cbits/polynomial.cpp
    cbits/polynomial_state.cpp
    cbits/random.cpp
    cbits/spin.cpp
    cbits/spin_basis.cpp
    cbits/pool.cpp
    cbits/trim.cpp
)
nqs_cbits_add_low_level_flags(_C_nqs)
target_compile_definitions(_C_nqs PUBLIC
    TORCH_API_INCLUDE_EXTENSION_H=1
    TORCH_EXTENSION_NAME="_C_nqs")
target_link_libraries(_C_nqs PRIVATE nqs_cbits_Common)
# target_compile_options(_C_nqs PRIVATE "-fsanitize=address")
# target_link_libraries(_C_nqs PRIVATE "-fsanitize=address")

add_subdirectory(cbits/test/headers)

install(TARGETS _C_nqs
        LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/nqs_playground)

