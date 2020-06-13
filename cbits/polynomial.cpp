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
TCM_EXPORT QuantumState::QuantumState() = default;

TCM_EXPORT auto QuantumState::operator+=(
    std::pair<bits512 const, std::complex<double>> const& item) -> QuantumState&
{
    TCM_CHECK(!_locked_table.has_value(), std::runtime_error,
              "table is frozen");
    _table.upsert(
        item.first,
        [c = item.second](std::complex<double>& value) { value += c; },
        item.second);
    return *this;
}

TCM_EXPORT auto QuantumState::clear() -> void
{
    TCM_CHECK(!_locked_table.has_value(), std::runtime_error,
              "table is frozen");
    _table.clear();
}

TCM_EXPORT auto QuantumState::freeze() -> void
{
    if (!_locked_table.has_value()) {
        _locked_table.emplace(_table.lock_table());
    }
}

TCM_EXPORT auto QuantumState::unfreeze() -> void
{
    if (_locked_table.has_value()) { _locked_table.reset(); }
}

TCM_EXPORT auto QuantumState::norm() const -> double
{
    TCM_CHECK(_locked_table.has_value(), std::runtime_error,
              "table is not frozen");
    auto norm = 0.0;
    for (auto const& item : unsafe_locked_table()) {
        norm += std::norm(item.second);
    }
    return norm;
}

TCM_EXPORT auto QuantumState::unsafe_locked_table() const
    -> table_type::locked_table const&
{
    TCM_CHECK(_locked_table.has_value(), std::runtime_error,
              "table is not frozen");
    return *_locked_table;
}

TCM_EXPORT auto swap(QuantumState& x, QuantumState& y) -> void
{
    x._table.swap(y._table);
    x._locked_table.swap(y._locked_table);
}

} // namespace v2

TCM_NAMESPACE_END
