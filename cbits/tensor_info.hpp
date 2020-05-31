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
#include <type_traits>
#include <utility>

#if __cplusplus >= 201402L
#    define TCM_CXX14_CONSTEXPR constexpr
#else
#    define TCM_CXX14_CONSTEXPR
#endif

TCM_NAMESPACE_BEGIN

namespace detail {
#if __cplusplus >= 201402L
using std::index_sequence;
using std::make_index_sequence;
#else

template <std::size_t...> struct index_sequence {};

template <std::size_t N, std::size_t... Rest>
struct make_index_sequence_impl
    : public make_index_sequence_impl<N - 1, N - 1, Rest...> {};

template <std::size_t... Indices>
struct make_index_sequence_impl<0, Indices...> {
    using type = index_sequence<Indices...>;
};

template <std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;
#endif
} // namespace detail

template <typename T, size_t Dims = 1, typename Index = int64_t>
struct TensorInfo {
    constexpr TensorInfo() noexcept
        : TensorInfo{detail::make_index_sequence<Dims>{}}
    {}

    constexpr TensorInfo(T* _data, Index const _sizes[Dims],
                         Index const _strides[Dims]) noexcept
        : TensorInfo{_data, _sizes, _strides,
                     detail::make_index_sequence<Dims>{}}
    {}

    template <class D = void, class = typename std::enable_if<
                                  std::is_same<D, D>::value && Dims == 1>::type>
    constexpr TensorInfo(T* _data, Index const _size,
                         Index const _stride) noexcept
        : data{_data}, sizes{_size}, strides{_stride}
    {}

    constexpr TensorInfo(TensorInfo const&) noexcept = default;
    constexpr TensorInfo(TensorInfo&&) noexcept      = default;
#if defined(                                                                   \
    TCM_NVCC) // For some reason constexpr and noexcept default assignments don't work with nvcc...
    TensorInfo& operator=(TensorInfo const&) = default;
    TensorInfo& operator=(TensorInfo&&) = default;
#else
    constexpr TensorInfo& operator=(TensorInfo const&) noexcept = default;
    constexpr TensorInfo& operator=(TensorInfo&&) noexcept = default;
#endif

    template <class D = void,
              class   = typename std::enable_if<
                  std::is_same<D, D>::value && !std::is_const<T>::value>::type>
    operator TensorInfo<T const, Dims, Index>() const noexcept
    {
        return {data, sizes, strides};
    }

  private:
    template <size_t... Is>
    constexpr TensorInfo(detail::index_sequence<Is...> /*unused*/) noexcept
        : data{nullptr}, sizes{(Is, Index{0})...}, strides{(Is, Index{0})...}
    {}

    template <size_t... Is>
    constexpr TensorInfo(T* _data, Index const _sizes[Dims],
                         Index const _strides[Dims],
                         detail::index_sequence<Is...> /*unused*/) noexcept
        : data{_data}, sizes{_sizes[Is]...}, strides{_strides[Is]...}
    {}

  public:
    template <class D = void, class = typename std::enable_if<
                                  std::is_same<D, D>::value && Dims == 1>::type>
    constexpr auto size() const noexcept -> Index
    {
        return sizes[0];
    }

    template <class D = void, class = typename std::enable_if<
                                  std::is_same<D, D>::value && Dims == 1>::type>
    constexpr auto stride() const noexcept -> Index
    {
        return strides[0];
    }

    template <class D = void, class = typename std::enable_if<
                                  std::is_same<D, D>::value && Dims == 1>::type>
    TCM_CXX14_CONSTEXPR auto operator[](Index i) const noexcept -> T
    {
        TCM_ASSERT(0 <= i && i < sizes[0], "index out of bounds");
        return data[i * strides[0]];
    }

    template <class D = void, class = typename std::enable_if<
                                  std::is_same<D, D>::value && Dims == 1>::type>
    TCM_CXX14_CONSTEXPR auto operator[](Index i) noexcept -> T&
    {
        TCM_ASSERT(0 <= i && i < sizes[0], "index out of bounds");
        return data[i * strides[0]];
    }

    T*    data;
    Index sizes[Dims];
    Index strides[Dims];
};

template <class T>
TCM_FORCEINLINE auto row(TensorInfo<T, 2> const& x, int64_t const i)
    -> TensorInfo<T, 1>
{
    TCM_ASSERT(0 <= i && i < x.sizes[0], "index out of bounds");
    return TensorInfo<T, 1>{x.data + i * x.strides[0], x.sizes[1],
                            x.strides[1]};
}

template <class T>
TCM_FORCEINLINE auto column(TensorInfo<T, 2> const& x, int64_t const i)
    -> TensorInfo<T, 1>
{
    TCM_ASSERT(0 <= i && i < x.sizes[1], "index out of bounds");
    return TensorInfo<T, 1>{x.data + i * x.strides[1], x.sizes[0],
                            x.strides[0]};
}

template <class T>
TCM_FORCEINLINE auto slice(TensorInfo<T> const& x, int64_t const start,
                           int64_t end = std::numeric_limits<int64_t>::max())
    -> TensorInfo<T>

{
    if (end == std::numeric_limits<int64_t>::max()) { end = x.size(); }
    TCM_ASSERT(0 <= start && start < x.size(), "index out of bounds");
    TCM_ASSERT(start <= end && end < x.size(), "index out of bounds");
    return TensorInfo<T>{x.data + start * x.stride(), end - start, x.stride()};
}

template <class T, bool Checks = true>
auto obtain_tensor_info(torch::Tensor x, char const* name = nullptr)
    -> TensorInfo<T>;

TCM_NAMESPACE_END
