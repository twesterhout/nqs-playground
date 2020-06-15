// Copyright (c) 2019-2020, Tom Westerhout
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

#include "polynomial.hpp"

TCM_NAMESPACE_BEGIN

namespace v2 {

TCM_EXPORT QuantumState::Part::Part(Part&&) noexcept : _table{}, _mutex{}
{
    detail::assert_fail("false", __FILE__, __LINE__, TCM_CURRENT_FUNCTION,
                        "Part(Part&&) should never be called");
}

TCM_EXPORT QuantumState::QuantumState() : _parts{}, _mask{16 - 1}
{
    _parts.reserve(16);
    for (auto i = 0; i < 16; ++i) {
        _parts.emplace_back();
    }
}

TCM_EXPORT auto QuantumState::operator+=(
    std::pair<bits512 const, std::complex<double>> const& item) -> QuantumState&
{
    auto const index = item.first.words[0] & _mask;
    auto&      part  = _parts.at(index);
    {
        std::lock_guard<std::mutex> guard{part._mutex};
        part._table[item.first] += item.second;
    }
    return *this;
}

TCM_EXPORT auto QuantumState::clear() -> void
{
    for (auto& part : _parts) {
        part._table.clear();
    }
}

TCM_EXPORT auto QuantumState::empty() const noexcept -> bool
{
    for (auto const& part : _parts) {
        if (!part._table.empty()) { return false; }
    }
    return true;
}

TCM_EXPORT auto QuantumState::norm() const -> double
{
    auto norm = 0.0;
    for (auto const& part : _parts) {
        for (auto const& item : part._table) {
            norm += std::norm(item.second);
        }
    }
    return norm;
}

TCM_EXPORT auto swap(QuantumState& x, QuantumState& y) -> void
{
    using std::swap;
    swap(x._parts, y._parts);
}
} // namespace v2

TCM_NAMESPACE_END
