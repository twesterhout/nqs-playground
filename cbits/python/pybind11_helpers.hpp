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

#include "../common/errors.hpp"
#include "../common/wrappers.hpp"
#include "trim.hpp"

#include <lattice_symmetries/lattice_symmetries.h>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace pybind11 {
namespace detail {
#if 0
    template <> struct type_caster<c10::Device> {
      private:
        c10::Device payload = c10::Device{c10::DeviceType::CPU};

      public:
        auto load(handle src, bool convert) -> bool;

        operator c10::Device() const { return payload; }

        static constexpr auto name            = _("torch.device");
        template <class T> using cast_op_type = std::remove_reference_t<T>;
    };

    template <> struct type_caster<c10::ScalarType> {
      private:
        c10::ScalarType payload = torch::kFloat32;

        static auto string_to_dtype(std::string const& name) -> std::optional<c10::ScalarType>;

      public:
        auto load(handle src, bool convert) -> bool;

        operator c10::ScalarType() const { return payload; }

        static constexpr auto name            = _("torch.dtype");
        template <class T> using cast_op_type = std::remove_reference_t<T>;
    };
#endif

    template <> struct type_caster<ls_bits512> {
      public:
        // This macro establishes the name 'bits512' in function signatures and declares a local
        // variable 'value' of type bits512
        PYBIND11_TYPE_CASTER(ls_bits512, _("ls_bits512"));

        // Conversion part 1 (Python->C++): convert a PyObject into a ls_bits512 instance or return
        // false upon failure. The second argument indicates whether implicit conversions should be
        // applied.
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
            auto mask = reinterpret_steal<object>(PyLong_FromUnsignedLongLong(0xFFFFFFFFFFFFFFFF));
            TCM_CHECK(mask.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            for (auto i = 0;; ++i) {
                auto word = reinterpret_steal<object>(PyNumber_And(number.ptr(), mask.ptr()));
                TCM_CHECK(word.ptr() != nullptr, std::runtime_error,
                          "Failed to perform a bitwise and Python integers. "
                          "This is probably a bug");
                value.words[i] = PyLong_AsUnsignedLongLong(word.ptr());
                if (i == 7) { break; }
                number =
                    reinterpret_steal<object>(PyNumber_InPlaceRshift(number.ptr(), shift.ptr()));
                TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                          "Failed to perform an in-place right shift. This is "
                          "probably a bug");
            }
            return true;
        }

        // Conversion part 2 (C++ -> Python): convert an bits512 instance into a Python object. The
        // second and third arguments are used to indicate the return value policy and parent object
        // (for ``return_value_policy::reference_internal``) and are generally ignored by implicit
        // casters.
        static auto cast(ls_bits512 const& src, return_value_policy /* policy */,
                         handle /* parent */) -> handle
        {
            auto shift = reinterpret_steal<object>(PyLong_FromLong(64L));
            TCM_CHECK(shift.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            auto number = reinterpret_steal<object>(PyLong_FromUnsignedLongLong(src.words[7]));
            TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                      "Failed to construct Python integer constant. This is "
                      "probably a bug");
            for (auto i = 7; i-- > 0;) { // Iterates from 6 to 0 (inclusive)
                number =
                    reinterpret_steal<object>(PyNumber_InPlaceLshift(number.ptr(), shift.ptr()));
                TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                          "Failed to perform an in-place left shift. This is "
                          "probably a bug");
                auto word = reinterpret_steal<object>(PyLong_FromUnsignedLongLong(src.words[i]));
                TCM_CHECK(word.ptr() != nullptr, std::runtime_error,
                          "Failed to construct Python integer constant. This "
                          "is probably a bug");
                number = reinterpret_steal<object>(PyNumber_InPlaceOr(number.ptr(), word.ptr()));
                TCM_CHECK(number.ptr() != nullptr, std::runtime_error,
                          "Failed to perform an in-place or. This is probably a bug");
            }
            return number.release();
        }
    };

    template <> struct type_caster<ls_operator> {
      private:
        ls_operator* payload;

      public:
        auto load(handle src, bool convert) -> bool
        {
            if (src.is_none()) { return false; }
            auto operator_type = module_::import("lattice_symmetries").attr("Operator");
            if (!isinstance(src, operator_type)) { return false; }
            payload =
                reinterpret_cast<ls_operator*>(src.attr("_payload").attr("value").cast<intptr_t>());
            TCM_ASSERT(payload != nullptr, "");
            return true;
        }

        operator ls_operator*() { return payload; }
        operator ls_operator&()
        {
            TCM_ASSERT(payload != nullptr, "");
            return *payload;
        }

        static constexpr auto name            = _("lattice_symmetries.Operator");
        template <class T> using cast_op_type = pybind11::detail::cast_op_type<T>;
    };

    template <> struct type_caster<ls_spin_basis> {
      private:
        ls_spin_basis* payload;

      public:
        auto load(handle src, bool convert) -> bool
        {
            if (src.is_none()) { return false; }
            auto basis_type = module_::import("lattice_symmetries").attr("SpinBasis");
            if (!isinstance(src, basis_type)) { return false; }
            payload = reinterpret_cast<ls_spin_basis*>(
                src.attr("_payload").attr("value").cast<intptr_t>());
            return true;
        }

        operator ls_spin_basis*() { return payload; }
        operator ls_spin_basis&() { return *payload; }

        static constexpr auto name            = _("lattice_symmetries.SpinBasis");
        template <class T> using cast_op_type = pybind11::detail::cast_op_type<T>;
    };

    template <> struct type_caster<::TCM_NAMESPACE::ForwardT> {
      public:
        auto load(handle src, bool convert) -> bool
        {
            if (src.is_none()) {
                // Defer accepting None to other overloads (if we aren't in convert mode):
                if (!convert) return false;
                return true;
            }

            // We are dealing with torch.jit.ScriptMethod
            if (type_caster<torch::jit::script::Method> method_caster;
                method_caster.load(src, convert)) {
                value = [f = cast_op<torch::jit::script::Method>(std::move(method_caster))](
                            auto x) mutable { return f({std::move(x)}).toTensor(); };
                return true;
            }

            // We are dealing with torch.jit.ScriptModule
            if (type_caster<torch::jit::script::Module> module_caster;
                module_caster.load(src, convert)) {
                auto module = cast_op<torch::jit::script::Module>(std::move(module_caster));
                auto method = module.get_method("forward");
                // This relies on the fact that Module and Method have reference
                // semantics.
                value = [parent = std::move(module), f = std::move(method)](auto x) mutable {
                    return f({std::move(x)}).toTensor();
                };
                return true;
            }

            // We are dealing with a Python function
            if (!isinstance<function>(src)) return false;
            auto func = reinterpret_borrow<function>(src);

            // The following idea is taken from pybind11/functional.h. It
            // ensures that GIL is held during functor destruction
            struct func_handle {
                function f;
                func_handle(function&& f_) : f{std::move(f_)} {}
                func_handle(func_handle&&) noexcept = default;
                func_handle(func_handle const& f_)
                {
                    gil_scoped_acquire acq;
                    f = f_.f;
                }
                auto operator=(func_handle&&) noexcept -> func_handle& = default;
                auto operator=(func_handle const&) -> func_handle& = delete;
                ~func_handle()
                {
                    gil_scoped_acquire acq;
                    function           kill_f(std::move(f));
                }
            };

            value = [handle = func_handle{std::move(func)}](auto x) {
                gil_scoped_acquire acq;
                auto               r = handle.f(std::move(x));
                return r.template cast<torch::Tensor>();
            };
            return true;
        }

        PYBIND11_TYPE_CASTER(::TCM_NAMESPACE::ForwardT, _("Callable[[Tensor], Tensor]"));
    };
} // namespace detail
} // namespace pybind11
