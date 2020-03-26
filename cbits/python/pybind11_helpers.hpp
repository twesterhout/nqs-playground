// Copyright (c) 2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../symmetry.hpp"

#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {
    template <> struct type_caster<::TCM_NAMESPACE::bits512> {
      public:
        // This macro establishes the name 'bits512' in
        // function signatures and declares a local variable
        // 'value' of type bits512
        PYBIND11_TYPE_CASTER(::TCM_NAMESPACE::bits512, _("bits512"));

        // Conversion part 1 (Python->C++): convert a PyObject into a bits512
        // instance or return false upon failure. The second argument
        // indicates whether implicit conversions should be applied.
        auto load(handle src, bool implicit) -> bool
        {
            /* Extract PyObject from handle */
            auto number = object{};
            if (implicit) {
                // Try converting the argument to a Python int
                number = reinterpret_steal<object>(PyNumber_Long(src.ptr()));
                if (number.ptr() == nullptr) { return false; }
            }
            else {
                // Just check whether the argument is an int
                if (!PyLong_Check(src.ptr())) { return false; }
                number = reinterpret_borrow<object>(src.ptr());
            }
            auto shift = reinterpret_steal<object>(PyLong_FromLong(64L));
            TCM_CHECK(shift.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            auto mask = reinterpret_steal<object>(
                PyLong_FromUnsignedLongLong(0xFFFFFFFFFFFFFFFF));
            TCM_CHECK(mask.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            for (auto i = 0;; ++i) {
                auto word = reinterpret_steal<object>(
                    PyNumber_And(number.ptr(), mask.ptr()));
                TCM_CHECK(word.ptr() != nullptr, std::runtime_error,
                          "Failed to perform a bitwise and Python integers. "
                          "This is probably a bug");
                value.words[i] = PyLong_AsUnsignedLongLong(word.ptr());
                if (i == 7) { break; }
                number = reinterpret_steal<object>(
                    PyNumber_InPlaceRshift(number.ptr(), shift.ptr()));
                TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                          "Failed to perform an in-place right shift. This is "
                          "probably a bug");
            }
            return true;
        }

        // Conversion part 2 (C++ -> Python): convert an bits512 instance into
        // a Python object. The second and third arguments are used to
        // indicate the return value policy and parent object (for
        // ``return_value_policy::reference_internal``) and are generally
        // ignored by implicit casters.
        static auto cast(::TCM_NAMESPACE::bits512 const& src,
                         return_value_policy /* policy */, handle /* parent */)
            -> handle
        {
            auto shift = reinterpret_steal<object>(PyLong_FromLong(64L));
            TCM_CHECK(shift.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            auto number = reinterpret_steal<object>(
                PyLong_FromUnsignedLongLong(src.words[7]));
            TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            for (auto i = 7; i-- > 0;) { // Iterates from 6 to 0 (inclusive)
                number = reinterpret_steal<object>(
                    PyNumber_InPlaceLshift(number.ptr(), shift.ptr()));
                TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                          "Failed to perform an in-place left shift. This is "
                          "probably a bug");
                auto word = reinterpret_steal<object>(
                    PyLong_FromUnsignedLongLong(src.words[i]));
                TCM_CHECK(word.ptr() != nullptr, std::runtime_error,
                          "Failed to construct Python integer constant. This "
                          "is probably a bug");
                number = reinterpret_steal<object>(
                    PyNumber_InPlaceOr(number.ptr(), word.ptr()));
                TCM_CHECK(
                    number.ptr() != nullptr, std::runtime_error,
                    "Failed to perform an in-place or. This is probably a bug");
            }
            return number.release();
        }
    };
} // namespace detail
} // namespace pybind11
