cmake_minimum_required(VERSION 3.9)
project(nqs_cbits)

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
        -Wstrict-overflow=1 -Wswitch-default -Wundef -Wno-unused)
endif()

set(PYBIND11_CPP_STANDARD "-std=c++14")
add_subdirectory(external/pybind11)

pybind11_add_module(_C_nqs MODULE SYSTEM cbits/nqs.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(_C_nqs PRIVATE -fsized-deallocation)
endif()
target_compile_options(_C_nqs PRIVATE ${TCM_WARNING_FLAGS})
target_include_directories(_C_nqs SYSTEM PRIVATE external)
