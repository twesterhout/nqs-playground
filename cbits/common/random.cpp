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

#include "random.hpp"
#include <random>
#include <omp.h>

TCM_NAMESPACE_BEGIN

namespace detail {
namespace {
    auto really_need_that_random_seed_now() -> uint64_t
    {
        std::random_device                      random_device;
        std::uniform_int_distribution<uint64_t> dist;
        auto const                              seed = dist(random_device);
        return seed;
    }
} // namespace
} // namespace detail

TCM_EXPORT auto global_random_generator() -> RandomGenerator&
{
    static thread_local RandomGenerator generator{detail::really_need_that_random_seed_now()};
    return generator;
}

TCM_EXPORT auto manual_seed(uint64_t const seed) -> void
{
#pragma omp parallel default(none) firstprivate(seed)
    {
        auto i = omp_get_thread_num();
        global_random_generator().seed(seed + static_cast<unsigned>(i));
    }
}

TCM_NAMESPACE_END
