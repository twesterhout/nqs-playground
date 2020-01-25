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

#include "bind_heisenberg.hpp"
#include "bind_metropolis.hpp"
#include "bind_polynomial_state.hpp"
#include "bind_spin_basis.hpp"
#include "bind_symmetry.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
PYBIND11_MODULE(_C, m)
{
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

    m.doc() = R"EOF()EOF";

    using namespace TCM_NAMESPACE;

    // bind_spin(m.ptr());
    bind_symmetry(m.ptr());
    bind_spin_basis(m.ptr());
    bind_heisenberg(m.ptr());
    // bind_heisenberg(m);
    // bind_explicit_state(m);
    // bind_polynomial(m);
    // bind_options(m);
    // bind_chain_result(m);
    // bind_sampling(m);
    // bind_networks(m);
    // bind_dataloader(m);
    bind_metropolis(m.ptr());
    bind_polynomial_state(m.ptr());
}
