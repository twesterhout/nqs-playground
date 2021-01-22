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

#include "bits512.hpp"
#include "errors.hpp"
#include <gsl/gsl-lite.hpp>
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
struct make_index_sequence_impl : public make_index_sequence_impl<N - 1, N - 1, Rest...> {};

template <std::size_t... Indices> struct make_index_sequence_impl<0, Indices...> {
    using type = index_sequence<Indices...>;
};

template <std::size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;
#endif
} // namespace detail

template <typename T, size_t Dims = 1, typename Index = int64_t> struct TensorInfo {
    constexpr TensorInfo() noexcept : TensorInfo{detail::make_index_sequence<Dims>{}} {}

    constexpr TensorInfo(T* _data, Index const _sizes[Dims], Index const _strides[Dims]) noexcept
        : TensorInfo{_data, _sizes, _strides, detail::make_index_sequence<Dims>{}}
    {}

    template <class D = void,
              class   = typename std::enable_if<std::is_same<D, D>::value && Dims == 1>::type>
    constexpr TensorInfo(T* _data, Index const _size, Index const _stride) noexcept
        : data{_data}, sizes{_size}, strides{_stride}
    {}

    template <class D = void,
              class   = typename std::enable_if<std::is_same<D, D>::value && Dims == 1>::type>
    /*implicit*/ constexpr TensorInfo(gsl::span<T> other) noexcept
        : data{other.data()}, sizes{other.size()}, strides{1}
    {}

    constexpr TensorInfo(TensorInfo const&) noexcept = default;
    constexpr TensorInfo(TensorInfo&&) noexcept      = default;
    // For some reason constexpr and noexcept default assignments don't work with nvcc...
#if defined(__CUDACC__)
    TensorInfo& operator=(TensorInfo const&) = default;
    TensorInfo& operator=(TensorInfo&&) = default;
#else
    constexpr TensorInfo& operator=(TensorInfo const&) noexcept = default;
    constexpr TensorInfo& operator=(TensorInfo&&) noexcept = default;
#endif

    template <class D = T, class = typename std::enable_if<!std::is_const<D>::value>::type>
    operator TensorInfo<D const, Dims, Index>() const noexcept
    {
        return {data, sizes, strides};
    }

    template <size_t D = Dims, class = typename std::enable_if<D == 1>::type>
    explicit operator gsl::span<T>() const
    {
        TCM_CHECK(stride() == 1, std::runtime_error,
                  fmt::format("cannot cast TensorInfo with stride {} to gsl::span", stride()));
        return gsl::span<T>{data, static_cast<uint64_t>(size())};
    }

    template <class D = void,
              class = typename std::enable_if<std::is_same<D, D>::value && !std::is_const<T>::value
                                              && Dims == 1>::type>
    explicit operator gsl::span<T const>() const
    {
        TCM_CHECK(stride() == 1, std::runtime_error,
                  fmt::format("cannot cast TensorInfo with stride {} to gsl::span", stride()));
        return gsl::span<T const>{data, static_cast<uint64_t>(size())};
    }

  private:
    template <size_t... Is>
    constexpr TensorInfo(detail::index_sequence<Is...> /*unused*/) noexcept
        : data{nullptr}, sizes{(Is, Index{0})...}, strides{(Is, Index{0})...}
    {}

    template <size_t... Is>
    constexpr TensorInfo(T* _data, Index const _sizes[Dims], Index const _strides[Dims],
                         detail::index_sequence<Is...> /*unused*/) noexcept
        : data{_data}, sizes{_sizes[Is]...}, strides{_strides[Is]...}
    {}

  public:
    template <int D = -1, class = typename std::enable_if<(D == -1 && Dims == 1)
                                                          || (D >= 0 && Dims >= 1)>::type>
    constexpr auto size() const noexcept -> Index
    {
        static_assert(D < static_cast<int>(Dims), "index out of bounds");
        constexpr auto i = D == -1 ? 0 : D;
        return sizes[i];
    }

    template <int D = -1, class = typename std::enable_if<(D == -1 && Dims == 1)
                                                          || (D >= 0 && Dims >= 1)>::type>
    constexpr auto stride() const noexcept -> Index
    {
        static_assert(D < static_cast<int>(Dims), "index out of bounds");
        constexpr auto i = D == -1 ? 0 : D;
        return strides[i];
    }

    template <class D = void,
              class   = typename std::enable_if<std::is_same<D, D>::value && Dims == 1>::type>
    TCM_CXX14_CONSTEXPR auto operator[](Index i) const noexcept -> T&
    {
        TCM_ASSERT(0 <= i && i < sizes[0], "index out of bounds");
        return data[i * strides[0]];
    }

    T*    data;
    Index sizes[Dims];
    Index strides[Dims];
};

template <class T>
TCM_FORCEINLINE auto row(TensorInfo<T, 2> const& x, int64_t const i) -> TensorInfo<T, 1>
{
    TCM_ASSERT(0 <= i && i < x.sizes[0], "index out of bounds");
    return TensorInfo<T, 1>{x.data + i * x.strides[0], x.sizes[1], x.strides[1]};
}

template <class T>
TCM_FORCEINLINE auto column(TensorInfo<T, 2> const& x, int64_t const i) -> TensorInfo<T, 1>
{
    TCM_ASSERT(0 <= i && i < x.sizes[1], "index out of bounds");
    return TensorInfo<T, 1>{x.data + i * x.strides[1], x.sizes[0], x.strides[0]};
}

template <class T>
TCM_FORCEINLINE auto slice(TensorInfo<T> const& x, int64_t const start,
                           int64_t end = std::numeric_limits<int64_t>::max()) -> TensorInfo<T>

{
    if (end == std::numeric_limits<int64_t>::max()) { end = x.size(); }
    TCM_ASSERT(0 <= start && start < x.size(), "index out of bounds");
    TCM_ASSERT(start <= end && end <= x.size(), "index out of bounds");
    return TensorInfo<T>{x.data + start * x.stride(), end - start, x.stride()};
}

namespace detail {
template <class T, size_t Dims, class = void> struct obtain_tensor_info_fn {
    auto operator()(torch::Tensor x, char const* name) const -> TensorInfo<T, Dims>
    {
        auto const arg_name = name != nullptr ? name : "tensor";
        auto const sizes    = x.sizes();
        TCM_CHECK(sizes.size() == Dims, std::invalid_argument,
                  fmt::format("{} has wrong shape: [{}]; expected a {}-dimensional tensor",
                              arg_name, fmt::join(sizes, ","), Dims));
        auto* data = x.data_ptr<T>();
        return {data, sizes.data(), x.strides().data()};
    }
};

template <class T, size_t Dims>
struct obtain_tensor_info_fn<T const, Dims> : public obtain_tensor_info_fn<T, Dims> {
    auto operator()(torch::Tensor const& x, char const* name) const -> TensorInfo<T const, Dims>
    {
        return static_cast<obtain_tensor_info_fn<T, Dims> const&>(*this)(x, name);
    }
};

template <size_t Dims>
struct obtain_tensor_info_fn<uint64_t, Dims> : public obtain_tensor_info_fn<int64_t, Dims> {
    auto operator()(torch::Tensor const& x, char const* name) const -> TensorInfo<uint64_t, Dims>
    {
        auto const i = static_cast<obtain_tensor_info_fn<int64_t, Dims> const&>(*this)(x, name);
        return {reinterpret_cast<uint64_t*>(i.data), i.sizes, i.strides};
    }
};

template <size_t Dims> struct obtain_tensor_info_fn<ls_bits512, Dims> {
    auto operator()(torch::Tensor const& x, char const* name) const -> TensorInfo<ls_bits512, Dims>
    {
        auto const arg_name         = name != nullptr ? name : "tensor";
        auto const sizes            = x.sizes();
        auto const original_strides = x.strides();
        TCM_CHECK(sizes.size() == Dims + 1, std::invalid_argument,
                  fmt::format("{} has wrong shape: [{}]; expected a {}-dimensional tensor",
                              arg_name, fmt::join(sizes, ","), Dims + 1));
        TCM_CHECK(
            original_strides.back() == 1, std::invalid_argument,
            fmt::format("{} must be contiguous along the last dimension, but has strides: [{}]",
                        arg_name, fmt::join(original_strides, ",")));
        auto* data = reinterpret_cast<ls_bits512*>(x.data_ptr<int64_t>());

        int64_t strides[Dims];
        std::transform(original_strides.data(), original_strides.data() + Dims, strides,
                       [](auto const k) {
                           TCM_CHECK(k % 8 == 0, std::runtime_error,
                                     "expected strides of 'x' to be multiples of 8");
                           return k / 8;
                       });
        return {data, sizes.data(), strides};
    }
};
} // namespace detail

template <class T, size_t Dims = 1>
auto tensor_info(torch::Tensor x, char const* name = nullptr) -> TensorInfo<T, Dims>
{
    return detail::obtain_tensor_info_fn<T, Dims>{}(std::move(x), name);
}

// template <class T>
// auto obtain_tensor_info(torch::Tensor x, char const* name = nullptr) -> TensorInfo<T>;

// template <class T, bool Checks = true>
// auto obtain_tensor_info(torch::Tensor x, char const* name = nullptr)
//     -> TensorInfo<T>;

TCM_NAMESPACE_END
