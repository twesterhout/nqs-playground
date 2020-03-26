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

#include "config.hpp"
#include <torch/types.h>
#include <utility>

TCM_NAMESPACE_BEGIN

template <typename T, size_t Dims = 1, typename Index = int64_t>
struct TensorInfo {
    constexpr TensorInfo() noexcept
        : TensorInfo{std::make_index_sequence<Dims>{}}
    {}

    constexpr TensorInfo(T* _data, Index const _sizes[Dims],
                         Index const _strides[Dims]) noexcept
        : TensorInfo{_data, _sizes, _strides, std::make_index_sequence<Dims>{}}
    {}

    constexpr TensorInfo(TensorInfo const&) noexcept = default;
    constexpr TensorInfo(TensorInfo&&) noexcept      = default;
    constexpr auto operator=(TensorInfo const&) noexcept
        -> TensorInfo&     = default;
    constexpr auto operator=(TensorInfo&&) noexcept -> TensorInfo& = default;

    template <class = std::enable_if_t<!std::is_const_v<T>>>
    operator TensorInfo<T const, Dims, Index>() const noexcept
    {
        return {data, sizes, strides};
    }

  private:
    template <size_t... Is>
    constexpr TensorInfo(std::index_sequence<Is...> /*unused*/) noexcept
        : data{nullptr}, sizes{(Is, Index{0})...}, strides{(Is, Index{0})...}
    {}

    template <size_t... Is>
    constexpr TensorInfo(T* _data, Index const _sizes[Dims],
                         Index const _strides[Dims],
                         std::index_sequence<Is...> /*unused*/) noexcept
        : data{_data}, sizes{_sizes[Is]...}, strides{_strides[Is]...}
    {}

  public:
    template <class = std::enable_if_t<Dims == 1>>
    constexpr auto size() const noexcept
    {
        return sizes[0];
    }

    template <class = std::enable_if_t<Dims == 1>>
    constexpr auto stride() const noexcept
    {
        return strides[0];
    }

    T*    data;
    Index sizes[Dims];
    Index strides[Dims];
};

template <class T, bool Checks = true>
auto obtain_tensor_info(torch::Tensor x, char const* name = nullptr)
    -> TensorInfo<T>;

TCM_NAMESPACE_END
