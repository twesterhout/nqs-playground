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

#include "common.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "random.hpp"

#include <boost/align/aligned_allocator.hpp>
#include <boost/detail/workaround.hpp>
#include <gsl/gsl-lite.hpp>
#include <torch/types.h>

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#    pragma GCC diagnostic ignored "-Wsign-promo"
#    pragma GCC diagnostic ignored "-Wswitch-default"
#    pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#    pragma GCC diagnostic ignored "-Wstrict-overflow"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#    pragma GCC diagnostic ignored "-Wshadow"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#    pragma clang diagnostic ignored "-Wsign-conversion"
#    pragma clang diagnostic ignored "-Wswitch-enum"
#    pragma clang diagnostic ignored "-Wundefined-func-template"
#endif
#include <pybind11/numpy.h>
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

#include <immintrin.h>
#include <random> // std::uniform_int_distribution

#if BOOST_WORKAROUND(BOOST_GCC, <= 80000)
// Taken from Intel's immintrin.h
#    define _mm256_set_m128(/* __m128 */ hi, /* __m128 */ lo)                  \
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 0x1)

#    define _mm256_set_m128d(/* __m128d */ hi, /* __m128d */ lo)               \
        _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), (hi), 0x1)

#    define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo)               \
        _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)

#    define _mm256_setr_m128(lo, hi) _mm256_set_m128((hi), (lo))
#    define _mm256_setr_m128d(lo, hi) _mm256_set_m128d((hi), (lo))
#    define _mm256_setr_m128i(lo, hi) _mm256_set_m128i((hi), (lo))
#endif

TCM_NAMESPACE_BEGIN

enum class Spin : unsigned char {
    down = 0x00,
    up   = 0x01,
};

namespace detail {
// TODO(twesterhout): Remove me!
auto spin_configuration_to_string(gsl::span<float const> spin) -> std::string;
} // namespace detail

struct IdentityProjection {
    template <class T> constexpr decltype(auto) operator()(T&& x) const noexcept
    {
        return std::forward<T>(x);
    }
};

template <class RandomAccessIterator, class Projection = IdentityProjection>
TCM_NOINLINE auto unpack_to_tensor(RandomAccessIterator first,
                                   RandomAccessIterator last, torch::Tensor dst,
                                   Projection proj = Projection{}) -> void;

// [SpinVector] {{{
class TCM_EXPORT SpinVector {

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#endif
    union {
        struct {
            uint16_t spin[7];
            uint16_t size;
        };
        __m128i as_ints;
    } _data;
    static_assert(sizeof(_data) == 16, "");
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

    class SpinReference;

  public:
    /// Constructs an empty spin configuration.
    constexpr SpinVector() noexcept : _data{}
    {
        TCM_ASSERT(is_valid(), "");
        // _data.as_ints[0] = 0;
        // _data.as_ints[1] = 0;
    }

    constexpr SpinVector(SpinVector const&) noexcept = default;
    constexpr SpinVector(SpinVector&&) noexcept      = default;
    constexpr SpinVector& operator=(SpinVector const&) noexcept = default;
    constexpr SpinVector& operator=(SpinVector&&) noexcept = default;

    explicit SpinVector(gsl::span<float const>);
    explicit SpinVector(gsl::span<float const>, UnsafeTag) TCM_NOEXCEPT;

    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    explicit SpinVector(pybind11::array_t<float, ExtraFlags> const& spins);
    explicit SpinVector(pybind11::str str);

    explicit SpinVector(torch::Tensor const& spins);
    explicit SpinVector(torch::TensorAccessor<float, 1> accessor);

    explicit constexpr SpinVector(unsigned size, uint64_t spins);

    template <class Generator>
    static auto random(unsigned size, int magnetisation, Generator& generator)
        -> SpinVector;

    template <class Generator>
    static auto random(unsigned size, Generator& generator) -> SpinVector;

    constexpr auto        size() const noexcept -> unsigned;
    static constexpr auto max_size() noexcept -> unsigned;

    /// Returns the magnetisation of the spin configuration.
    ///
    /// \note Not constexpr, because of the use of intrinsics
    inline /*constexpr*/ auto magnetisation() const noexcept -> int;

    constexpr auto operator[](unsigned const i) const & TCM_NOEXCEPT -> Spin;
    constexpr auto operator[](unsigned const i) && TCM_NOEXCEPT -> Spin;
    constexpr auto operator[](unsigned const i) & TCM_NOEXCEPT -> SpinReference;

    constexpr auto at(unsigned const i) const& -> Spin;
    constexpr auto at(unsigned const i) && -> Spin;
    constexpr auto at(unsigned const i) & -> SpinReference;

    /// Flips the `i`'th spin.
    constexpr auto flip(unsigned i) TCM_NOEXCEPT -> void;

    /// Returns a new spin configuration with spins at `indices` flipped.
    template <size_t N>
    constexpr auto flipped(std::array<unsigned, N> indices) const TCM_NOEXCEPT
        -> SpinVector;

    constexpr auto
    flipped(std::initializer_list<unsigned> indices) const TCM_NOEXCEPT
        -> SpinVector;

    /// Compares spin configurations for equality.
    ///
    /// Only SpinVectors of the same length can be compared.
    inline auto operator==(SpinVector const& other) const TCM_NOEXCEPT -> bool;
    inline auto operator!=(SpinVector const& other) const TCM_NOEXCEPT -> bool;

    // TODO(twesterhout): Remove this!
    inline auto operator<(SpinVector const& other) const TCM_NOEXCEPT -> bool;

    inline auto     hash() const noexcept -> size_t;
    inline explicit operator uint64_t() const;
    inline explicit operator std::string() const;

    inline auto copy_to(gsl::span<float> buffer) const TCM_NOEXCEPT -> void;

    auto numpy() const -> pybind11::array_t<float, pybind11::array::c_style>;
    auto tensor() const -> torch::Tensor;

    constexpr auto key(UnsafeTag) const TCM_NOEXCEPT -> int64_t
    {
        TCM_ASSERT(size() <= 64, "Chain too long");
        return _data.spin[0];
    }

    constexpr auto is_valid() const TCM_NOEXCEPT -> bool
    {
        for (auto i = size(); i < max_size(); ++i) {
            if (unsafe_at(i) != Spin::down) { return false; }
        }
        return true;
    }

    template <class RandomAccessIterator, class Projection>
    TCM_NOINLINE friend auto
    unpack_to_tensor(RandomAccessIterator first, RandomAccessIterator last,
                     torch::Tensor dst, Projection proj) -> void;

    static auto numpy_dtype() -> pybind11::dtype;

  private:
    // [private junk] {{{
    static constexpr auto get_bit(uint16_t const x,
                                  unsigned const i) TCM_NOEXCEPT -> unsigned
    {
        TCM_ASSERT(i < 16u, "Index out of bounds");
        return (static_cast<unsigned>(x) >> (15u - i)) & 1u;
    }

    static constexpr auto flip_bit(uint16_t& x, unsigned const i) TCM_NOEXCEPT
        -> void
    {
        TCM_ASSERT(i < 16u, "Index out of bounds");
        x ^= static_cast<uint16_t>(1u << (15u - i));
    }

    static constexpr auto set_bit(uint16_t& x, unsigned const i,
                                  Spin const spin) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(i < 16, "Index out of bounds");
        x = (x & ~(1u << (15u - i)))
            | static_cast<uint16_t>(static_cast<unsigned>(spin) << (15u - i));
    }

    /// Returns whether `x` represents a valid spin (i.e. `x == 1` or `x == -1`)
    static constexpr auto is_valid_spin(float const x) noexcept -> bool
    {
        return x == -1.0f || x == 1.0f;
    }

    /// Returns whether all elements of `xs` represent valid spins.
    static auto is_valid_spin(__m256 const xs) noexcept -> bool
    {
        return _mm256_movemask_ps(_mm256_or_ps(
                   _mm256_cmp_ps(xs, _mm256_set1_ps(1.0f), _CMP_EQ_OQ),
                   _mm256_cmp_ps(xs, _mm256_set1_ps(-1.0f), _CMP_EQ_OQ)))
               == 0xFF;
    }

    /// Returns whether all elements of the given range represent valid spins.
    static auto is_valid_spin(gsl::span<float const> const range) noexcept
        -> bool
    {
        constexpr auto vector_size = size_t{8};
        auto const     chunks      = range.size() / vector_size;
        auto const     rest        = range.size() % vector_size;
        auto           chunks_good = true;
        auto const     data        = range.data();
        for (auto i = size_t{0}; i < chunks; ++i) {
            chunks_good =
                chunks_good
                && is_valid_spin(_mm256_loadu_ps(data + i * vector_size));
        }
        auto rest_good = true;
        for (auto i = size_t{0}; i < rest; ++i) {
            rest_good =
                rest_good && is_valid_spin(data[chunks * vector_size + i]);
        }
        return chunks_good && rest_good;
    }

    /// Checks that the range represents a valid spin configuration.
    static auto check_range(gsl::span<float const> const range) -> void
    {
        TCM_CHECK(range.size() <= max_size(), std::overflow_error,
                  fmt::format("range too long: {}; expected <={}", range.size(),
                              max_size()));
        TCM_CHECK(is_valid_spin(range), std::domain_error,
                  fmt::format("invalid spin configuration {}; every spin must "
                              "be either -1 or 1",
                              detail::spin_configuration_to_string(range)));
    }

    /// An overload of check_range that does the checking only in Debug builds.
    static auto check_range(gsl::span<float const> const range,
                            UnsafeTag) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(range.size() <= max_size(), "Spin chain too long");
        TCM_ASSERT(is_valid_spin(range), "Invalid spin configuration");
        static_cast<void>(range);
    }

    TCM_FORCEINLINE static auto
    load_u16_short(gsl::span<float const> buffer) TCM_NOEXCEPT -> uint16_t
    {
        TCM_ASSERT(buffer.size() < 16, "Range too long");
        TCM_ASSERT(is_valid_spin(buffer), "Invalid spin value");
        auto result = uint16_t{0};
        for (auto i = 0u; i < buffer.size(); ++i) {
            set_bit(result, i, buffer[i] == 1.0f ? Spin::up : Spin::down);
        }
        return result;
    }

    TCM_FORCEINLINE static auto load_u16(__m256 const p0,
                                         __m256 const p1) TCM_NOEXCEPT
        -> uint16_t
    {
        TCM_ASSERT(is_valid_spin(p0), "Invalid spin value");
        TCM_ASSERT(is_valid_spin(p1), "Invalid spin value");
        auto const mask0 =
            _mm256_set_ps((1 << 8), (1 << 9), (1 << 10), (1 << 11), (1 << 12),
                          (1 << 13), (1 << 14), (1 << 15));
        auto const mask1 =
            _mm256_set_ps((1 << 0), (1 << 1), (1 << 2), (1 << 3), (1 << 4),
                          (1 << 5), (1 << 6), (1 << 7));
        auto const v0 = _mm256_cmp_ps(p0, _mm256_set1_ps(1.0f), _CMP_EQ_OQ);
        auto const v1 = _mm256_cmp_ps(p1, _mm256_set1_ps(1.0f), _CMP_EQ_OQ);
        return static_cast<uint16_t>(detail::hadd(
            _mm256_add_ps(_mm256_and_ps(v0, mask0), _mm256_and_ps(v1, mask1))));
    }

    auto copy_from(gsl::span<float const> buffer) TCM_NOEXCEPT -> void
    {
        _data.as_ints     = _mm_set1_epi32(0);
        _data.size        = static_cast<uint16_t>(buffer.size());
        auto const chunks = buffer.size() / 16;
        auto const rest   = buffer.size() % 16;
        auto       data   = buffer.data();
        for (auto i = size_t{0}; i < chunks; ++i, data += 16) {
            _data.spin[i] =
                load_u16(_mm256_loadu_ps(data), _mm256_loadu_ps(data + 8));
        }
        if (rest != 0) {
            if (buffer.size() >= 16) {
                data -= (16u - rest);
                _data.spin[chunks] = static_cast<uint16_t>(
                    load_u16(_mm256_loadu_ps(data), _mm256_loadu_ps(data + 8))
                    << (16u - rest));
            }
            else {
                _data.spin[chunks] = load_u16_short({data, rest});
            }
        }
    }

  public:
    auto find_nth_up(unsigned n) const TCM_NOEXCEPT -> unsigned
    {
        ++n;
        auto i = 0u;
        while (true) {
            if (unsafe_at(i) == Spin::up) {
                if (--n == 0) { break; }
            }
            ++i;
            TCM_ASSERT(i < size(), "");
        }
        return i;
    }

    auto find_nth_down(unsigned n) const TCM_NOEXCEPT -> unsigned
    {
        ++n;
        auto i = 0u;
        while (true) {
            if (unsafe_at(i) == Spin::down) {
                if (--n == 0) { break; }
            }
            ++i;
            TCM_ASSERT(i < size(), "");
        }
        return i;
    }

  private:
    class SpinReference {
      public:
        constexpr SpinReference(uint16_t& ref, unsigned const n) TCM_NOEXCEPT
            : _ref{ref}
            , _i{n}
        {
            TCM_ASSERT(n < 16, "Index out of bounds.");
        }

        constexpr SpinReference(SpinReference const&) noexcept = default;
        constexpr SpinReference(SpinReference&&) noexcept      = default;
        SpinReference& operator=(SpinReference&&) = delete;
        SpinReference& operator=(SpinReference const&) = delete;

        SpinReference& operator=(Spin const spin) TCM_NOEXCEPT
        {
            set_bit(_ref, _i, spin);
            return *this;
        }

        constexpr operator Spin() const TCM_NOEXCEPT
        {
            return static_cast<Spin>(get_bit(_ref, _i));
        }

      private:
        uint16_t& _ref;
        unsigned  _i;
    };

    constexpr auto unsafe_at(unsigned const i) TCM_NOEXCEPT -> SpinReference
    {
        return SpinReference{_data.spin[i / 16u], i % 16u};
    }

    constexpr auto unsafe_at(unsigned const i) const TCM_NOEXCEPT -> Spin
    {
        return static_cast<Spin>(get_bit(_data.spin[i / 16u], i % 16u));
    }
    // }}}
};

static_assert(std::is_trivially_copyable<SpinVector>::value, "");
static_assert(std::is_trivially_destructible<SpinVector>::value, "");
// [SpinVector] }}}

// [SpinVector.implementation] {{{
constexpr SpinVector::SpinVector(unsigned size, uint64_t spins) : _data{}
{
    TCM_CHECK(size <= 64, std::overflow_error,
              fmt::format("size does not match the amount of data: {} > {}",
                          size, 8 * sizeof(size_t)));
    _data.size = size;
    spins <<= 64 - size;
    auto const chunks = (size + 7) / 8;
    // clang-format off
    switch (chunks) {
    case 8: _data.spin[3] |= (spins & 0x00000000000000FF) >> 0; TCM_FALLTHROUGH;
    case 7: _data.spin[3] |= (spins & 0x000000000000FF00) >> 0; TCM_FALLTHROUGH;
    case 6: _data.spin[2] |= (spins & 0x0000000000FF0000) >> 16; TCM_FALLTHROUGH;
    case 5: _data.spin[2] |= (spins & 0x00000000FF000000) >> 16; TCM_FALLTHROUGH;
    case 4: _data.spin[1] |= (spins & 0x000000FF00000000) >> 32; TCM_FALLTHROUGH;
    case 3: _data.spin[1] |= (spins & 0x0000FF0000000000) >> 32; TCM_FALLTHROUGH;
    case 2: _data.spin[0] |= (spins & 0x00FF000000000000) >> 48; TCM_FALLTHROUGH;
    case 1: _data.spin[0] |= (spins & 0xFF00000000000000) >> 48; TCM_FALLTHROUGH;
    default: TCM_FALLTHROUGH;
    } // end switch
    // clang-format on
}

constexpr auto SpinVector::size() const noexcept -> unsigned
{
    return _data.size;
}

constexpr auto SpinVector::max_size() noexcept -> unsigned
{
    return 8 * sizeof(_data.spin);
}

constexpr auto SpinVector::operator[](unsigned const i) const
    & TCM_NOEXCEPT -> Spin
{
    TCM_ASSERT(i < size(), "index out of bounds");
    return unsafe_at(i);
}

constexpr auto SpinVector::operator[](unsigned const i) && TCM_NOEXCEPT -> Spin
{
    TCM_ASSERT(i < size(), "index out of bounds");
    return unsafe_at(i);
}

constexpr auto SpinVector::operator[](unsigned const i)
    & TCM_NOEXCEPT -> SpinReference
{
    TCM_ASSERT(i < size(), "index out of bounds");
    return unsafe_at(i);
}

constexpr auto SpinVector::at(unsigned const i) const& -> Spin
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of bounds {}; expected <={}", i, size()));
    return unsafe_at(i);
}

constexpr auto SpinVector::at(unsigned const i) && -> Spin
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of bounds {}; expected <={}", i, size()));
    return unsafe_at(i);
}

constexpr auto SpinVector::at(unsigned const i) & -> SpinReference
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of bounds {}; expected <={}", i, size()));
    return unsafe_at(i);
}

constexpr auto SpinVector::flip(unsigned const i) TCM_NOEXCEPT -> void
{
    TCM_ASSERT(i < size(), "Index out of bounds.");
    auto const chunk = i / 16u;
    auto const rest  = i % 16u;
    flip_bit(_data.spin[chunk], rest);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated.");
}

template <size_t N>
constexpr auto
SpinVector::flipped(std::array<unsigned, N> is) const TCM_NOEXCEPT -> SpinVector
{
    SpinVector temp{*this};
    TCM_ASSERT(temp.is_valid(), "Bug! Copy constructor is broken.");
    for (auto const i : is) {
        temp.flip(i);
    }
    TCM_ASSERT(temp.is_valid(), "Bug! Post-condition violated.");
    return temp;
}

constexpr auto
SpinVector::flipped(std::initializer_list<unsigned> is) const TCM_NOEXCEPT
    -> SpinVector
{
    SpinVector temp{*this};
    TCM_ASSERT(temp.is_valid(), "Bug! Copy constructor is broken.");
    for (auto const i : is) {
        temp.flip(i);
    }
    TCM_ASSERT(temp.is_valid(), "Bug! Post-condition violated.");
    return temp;
}

inline auto SpinVector::magnetisation() const noexcept -> int
{
    static_assert(sizeof(unsigned long) == sizeof(uint64_t),
                  "\n" TCM_BUG_MESSAGE);
    static auto const size_mask = htobe64(0xFFFFFFFFFFFF0000);
    auto const        number_ones =
        __builtin_popcountll(static_cast<uint64_t>(_data.as_ints[0]))
        + __builtin_popcountll(static_cast<uint64_t>(_data.as_ints[1])
                               & size_mask);
    return 2 * number_ones - static_cast<int>(size());
}

inline auto SpinVector::operator==(SpinVector const& other) const TCM_NOEXCEPT
    -> bool
{
    TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
    TCM_ASSERT(other.is_valid(), "SpinVector is in an invalid state");
    return _mm_movemask_epi8(_data.as_ints == other._data.as_ints) == 0xFFFF;
}

inline auto SpinVector::operator!=(SpinVector const& other) const TCM_NOEXCEPT
    -> bool
{
    TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
    TCM_ASSERT(other.is_valid(), "SpinVector is in an invalid state");
    return _mm_movemask_epi8(_data.as_ints == other._data.as_ints) != 0xFFFF;
}

inline auto SpinVector::operator<(SpinVector const& other) const TCM_NOEXCEPT
    -> bool
{
    TCM_ASSERT(size() == other.size(),
               "Only equally-sized SpinVectors can be compared");
    TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
    TCM_ASSERT(other.is_valid(), "SpinVector is in an invalid state");
    return _data.as_ints[0] < other._data.as_ints[0];
}

inline auto SpinVector::hash() const noexcept -> size_t
{
    TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
    static_assert(sizeof(_data.as_ints[0]) == sizeof(size_t), "");

    auto const hash_uint64 = [](uint64_t x) noexcept->uint64_t
    {
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB;
        x = x ^ (x >> 31);
        return x;
    };

    auto const hash_combine = [hash_uint64](uint64_t seed,
                                            uint64_t x) noexcept->uint64_t
    {
        seed ^=
            hash_uint64(x) + uint64_t{0x9E3779B9} + (seed << 6) + (seed >> 2);
        return seed;
    };

    return hash_combine(hash_uint64(static_cast<uint64_t>(_data.as_ints[0])),
                        hash_uint64(static_cast<uint64_t>(_data.as_ints[1])));
}

inline SpinVector::operator uint64_t() const
{
    TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
    TCM_CHECK(size() <= 64, std::overflow_error,
              fmt::format("spin chain is too long to be converted to a 64-bit "
                          "int: {}; expected <=64",
                          size()));
    auto const x = (static_cast<uint64_t>(_data.spin[0]) << 48u)
                   + (static_cast<uint64_t>(_data.spin[1]) << 32u)
                   + (static_cast<uint64_t>(_data.spin[2]) << 16u)
                   + static_cast<uint64_t>(_data.spin[3]);
    return x >> (64u - size());
}

inline SpinVector::operator std::string() const
{
    std::string s(size(), 'X');
    for (auto i = 0u; i < size(); ++i) {
        s[i] = ((*this)[i] == Spin::up) ? '1' : '0';
    }
    return s;
}

namespace detail {

inline auto unpack(uint8_t const src) noexcept -> __m256
{
    auto const mask_high  = _mm_set_epi32(128, 64, 32, 16);
    auto const mask_low   = _mm_set_epi32(8, 4, 2, 1);
    auto const mask_final = _mm_set1_epi32(2);
    auto const x          = _mm_set1_epi32(src);

    auto low  = _mm_srai_epi32(_mm_mullo_epi32(mask_low, x), 6);
    low       = _mm_and_si128(low, mask_final);
    auto high = _mm_srai_epi32(_mm_mullo_epi32(mask_high, x), 6);
    high      = _mm_and_si128(high, mask_final);

    auto y = _mm256_cvtepi32_ps(_mm256_setr_m128i(low, high));
    y      = _mm256_sub_ps(y, _mm256_set1_ps(1.0f));

    // Tests the correctness
    TCM_ASSERT(([y, src]() {
                   alignas(32) float dst[8];
                   _mm256_store_ps(dst, y);
                   // This is how unpack is supposed to work
                   auto const get = [](uint8_t v, auto _i) -> float {
                       return 2.0f * static_cast<float>((v >> (7 - _i)) & 0x01)
                              - 1.0f;
                   };
                   for (auto i = 0; i < 8; ++i) {
                       if (get(src, i) != dst[i]) { return false; }
                   }
                   return true;
               }()),
               "");
    return y;
}

inline auto unpack(uint16_t const src, float* dst) noexcept -> void
{
    auto const mask_high  = _mm_set_epi32(128, 64, 32, 16);
    auto const mask_low   = _mm_set_epi32(8, 4, 2, 1);
    auto const mask_final = _mm_set1_epi32(2);

    auto const x_1 = _mm_set1_epi32(src >> 8);
    auto const x_2 = _mm_set1_epi32(src & 0xFF);

    auto low_1  = _mm_srai_epi32(_mm_mullo_epi32(mask_low, x_1), 6);
    auto high_1 = _mm_srai_epi32(_mm_mullo_epi32(mask_high, x_1), 6);
    auto low_2  = _mm_srai_epi32(_mm_mullo_epi32(mask_low, x_2), 6);
    auto high_2 = _mm_srai_epi32(_mm_mullo_epi32(mask_high, x_2), 6);
    low_1       = _mm_and_si128(low_1, mask_final);
    high_1      = _mm_and_si128(high_1, mask_final);
    low_2       = _mm_and_si128(low_2, mask_final);
    high_2      = _mm_and_si128(high_2, mask_final);

    auto y_1 = _mm256_cvtepi32_ps(_mm256_setr_m128i(low_1, high_1));
    auto y_2 = _mm256_cvtepi32_ps(_mm256_setr_m128i(low_2, high_2));
    y_1      = _mm256_sub_ps(y_1, _mm256_set1_ps(1.0f));
    y_2      = _mm256_sub_ps(y_2, _mm256_set1_ps(1.0f));
    _mm256_storeu_ps(dst, y_1);
    _mm256_storeu_ps(dst + 8, y_2);

    // Tests the correctness
    TCM_ASSERT(([src, dst]() {
                   // This is how unpack is supposed to work
                   auto const get = [](uint16_t v, auto _i) -> float {
                       return 2.0f * static_cast<float>((v >> (15 - _i)) & 0x01)
                              - 1.0f;
                   };
                   for (auto i = 0; i < 16; ++i) {
                       if (get(src, i) != dst[i]) { return false; }
                   }
                   return true;
               }()),
               "");
}

inline auto get_store_mask_for(unsigned const rest) TCM_NOEXCEPT -> __m256i
{
    // clang-format off
    __m256i const masks[9] = {
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0,  0),
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0,  0, -1),
        _mm256_set_epi32( 0,  0,  0,  0,  0,  0, -1, -1),
        _mm256_set_epi32( 0,  0,  0,  0,  0, -1, -1, -1),
        _mm256_set_epi32( 0,  0,  0,  0, -1, -1, -1, -1),
        _mm256_set_epi32( 0,  0,  0, -1, -1, -1, -1, -1),
        _mm256_set_epi32( 0,  0, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32( 0, -1, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
    };
    // clang-format on
    TCM_ASSERT(rest <= 8, "Invalid value for `rest`");
    return masks[rest];
}
} // namespace detail

inline auto
SpinVector::copy_to(gsl::span<float> const buffer) const TCM_NOEXCEPT -> void
{
    TCM_ASSERT(buffer.size() == size(), "Wrong buffer size");
    auto const chunks_16 = size() / 16;
    auto const rest_16   = size() % 16;
    auto const rest_8    = size() % 8;

    auto* p = buffer.data();
    auto  i = 0u;
    for (; i < chunks_16; ++i, p += 16) {
        detail::unpack(_data.spin[i], p);
    }

    if (rest_16 != 0) {
        if (rest_16 > 8) {
            auto const store_mask = detail::get_store_mask_for(rest_8);
            _mm256_storeu_ps(p, detail::unpack(_data.spin[i] >> 8));
            _mm256_maskstore_ps(p + 8, store_mask,
                                detail::unpack(_data.spin[i] & 0xFF));
        }
        else if (rest_16 == 8) {
            _mm256_storeu_ps(p, detail::unpack(_data.spin[i] >> 8));
        }
        else {
            auto const store_mask = detail::get_store_mask_for(rest_8);
            _mm256_maskstore_ps(p, store_mask,
                                detail::unpack(_data.spin[i] >> 8));
        }
    }

#if 0
    if (rest_16 != 0) {
        auto const store_mask = detail::get_store_mask_for(rest_8);
        if (rest_16 > 8) {
            _mm256_storeu_ps(p, detail::unpack(_data.spin[i] >> 8));
            _mm256_maskstore_ps(p + 8, store_mask,
                                detail::unpack(_data.spin[i] & 0xFF));
        }
        else {
            _mm256_maskstore_ps(p, store_mask,
                                detail::unpack(_data.spin[i] >> 8));
        }
    }
#endif

    // Testing correctness
    TCM_ASSERT(([this, buffer]() {
                   auto spin2float = [](Spin const s) noexcept->float
                   {
                       return s == Spin::up ? 1.0f : -1.0f;
                   };
                   for (auto k = 0u; k < size(); ++k) {
                       if (buffer[k] != spin2float((*this)[k])) {
                           return false;
                       }
                   }
                   return true;
               }()),
               "");
}

inline SpinVector::SpinVector(gsl::span<float const> buffer,
                              UnsafeTag /*unused*/) TCM_NOEXCEPT
{
    check_range(buffer, unsafe_tag);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

template <int ExtraFlags, class>
TCM_NOINLINE
SpinVector::SpinVector(pybind11::array_t<float, ExtraFlags> const& spins)
{
    TCM_CHECK_DIM(spins.ndim(), 1);
    auto buffer = gsl::span<float const>{spins.data(),
                                         static_cast<size_t>(spins.shape(0))};
    check_range(buffer);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

inline SpinVector::SpinVector(torch::TensorAccessor<float, 1> const accessor)
{
    TCM_CHECK(accessor.stride(0) == 1, std::invalid_argument,
              "input tensor must be contiguous");
    auto buffer = gsl::span<float const>{accessor.data(),
                                         static_cast<size_t>(accessor.size(0))};
    check_range(buffer);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

template <class Generator>
TCM_NOINLINE auto SpinVector::random(unsigned const size, Generator& generator)
    -> SpinVector
{
    TCM_CHECK(size <= SpinVector::max_size(), std::invalid_argument,
              fmt::format("invalid size {}; expected <={}", size,
                          SpinVector::max_size()));
    using Dist = std::uniform_int_distribution<uint16_t>;

    auto const chunks = size / 16u;
    auto const rest   = size % 16u;
    SpinVector spin;
    Dist       dist;
    for (unsigned i = 0u; i < chunks; ++i) {
        spin._data.spin[i] = dist(generator);
    }

    if (rest != 0) {
        TCM_ASSERT(rest < 16, "");
        using Param             = Dist::param_type;
        spin._data.spin[chunks] = static_cast<uint16_t>(
            dist(generator, Param{0, static_cast<uint16_t>((1 << rest) - 1)})
            << (16 - rest));
    }

    spin._data.size = static_cast<uint16_t>(size);
    TCM_ASSERT(spin.is_valid(), "Bug! Post-condition violated");
    return spin;
}

template <class Generator>
TCM_NOINLINE auto SpinVector::random(unsigned const size,
                                     int const      magnetisation,
                                     Generator&     generator) -> SpinVector
{
    TCM_CHECK(size <= SpinVector::max_size(), std::invalid_argument,
              fmt::format("invalid size {}; expected <={}", size,
                          SpinVector::max_size()));
    TCM_CHECK(
        static_cast<unsigned>(std::abs(magnetisation)) <= size,
        std::invalid_argument,
        fmt::format("magnetisation exceeds the number of spins: |{}| > {}",
                    magnetisation, size));
    TCM_CHECK((static_cast<int>(size) + magnetisation) % 2 == 0,
              std::runtime_error,
              fmt::format("{} spins cannot have a magnetisation of {}. `size + "
                          "magnetisation` must be even",
                          size, magnetisation));
    float      buffer[SpinVector::max_size()];
    auto const spin = gsl::span<float>{buffer, size};
    auto const number_ups =
        static_cast<size_t>((static_cast<int>(size) + magnetisation) / 2);
    auto const middle = std::begin(spin) + number_ups;
    std::fill(std::begin(spin), middle, 1.0f);
    std::fill(middle, std::end(spin), -1.0f);
    std::shuffle(std::begin(spin), std::end(spin), generator);
    auto compact_spin = SpinVector{spin};
    TCM_ASSERT(compact_spin.magnetisation() == magnetisation, "");
    return compact_spin;
}
// }}}

auto all_spins(unsigned n, optional<int> magnetisation)
    -> std::vector<SpinVector,
                   boost::alignment::aligned_allocator<SpinVector, 64>>;

// [unpack_to_tensor] {{{
template <class RandomAccessIterator, class Projection>
auto unpack_to_tensor(RandomAccessIterator first, RandomAccessIterator last,
                      torch::Tensor dst, Projection proj) -> void
{
    if (first == last) { return; }
    TCM_ASSERT(last - first > 0, "Invalid range");
    auto const size         = static_cast<size_t>(last - first);
    auto const number_spins = proj(*first).size();
    TCM_ASSERT(std::all_of(first, last,
                           [number_spins, &proj](auto& x) {
                               return proj(x).size() == number_spins;
                           }),
               "Input range contains variable size spin chains");
    TCM_ASSERT(dst.dim() == 2, fmt::format("Invalid dimension {}", dst.dim()));
    TCM_ASSERT(size == static_cast<size_t>(dst.size(0)),
               fmt::format("Sizes don't match: size={}, dst.size(0)={}", size,
                           dst.size(0)));
    TCM_ASSERT(static_cast<int64_t>(number_spins) == dst.size(1),
               fmt::format("Sizes don't match: number_spins={}, dst.size(1)={}",
                           number_spins, dst.size(1)));
    TCM_ASSERT(dst.is_contiguous(), "Output tensor must be contiguous");

    auto const chunks_16     = number_spins / 16;
    auto const rest_16       = number_spins % 16;
    auto const rest_8        = number_spins % 8;
    auto const copy_cheating = [chunks = chunks_16 + (rest_16 != 0)](
                                   SpinVector const& spin, float* out) {
        for (auto i = 0u; i < chunks; ++i, out += 16) {
            detail::unpack(spin._data.spin[i], out);
        }
    };

    auto const tail =
        std::min(((16UL - rest_16) + number_spins - 1) / number_spins, size);
    auto* data = dst.data_ptr<float>();
    auto  iter = first;
    for (auto i = size_t{0}; i < size - tail;
         ++i, ++iter, data += number_spins) {
        copy_cheating(proj(*iter), data);
    }
    for (auto i = size - tail; i < size; ++i, ++iter, data += number_spins) {
        proj(*iter).copy_to({data, number_spins});
    }

    TCM_ASSERT(([first, &dst, &proj]() {
                   auto accessor = dst.accessor<float, 2>();
                   auto _iter     = first;
                   for (auto i = int64_t{0}; i < accessor.size(0);
                        ++i, ++_iter) {
                       auto const s1 = proj(*_iter);
                       auto const s2 = SpinVector{accessor[i]};
                       if (s1 != s2) { return false; }
                   }
                   return true;
               }()),
               "");
}

template <class RandomAccessIterator, class Projection = IdentityProjection>
auto unpack_to_tensor(RandomAccessIterator first, RandomAccessIterator last,
                      Projection proj = Projection{}) -> torch::Tensor
{
    if (first == last) { return detail::make_tensor<float>(0); }
    TCM_ASSERT(last - first > 0, "Invalid range");
    auto const size         = static_cast<size_t>(last - first);
    auto const number_spins = proj(*first).size();
    auto       out          = detail::make_tensor<float>(size, number_spins);
    unpack_to_tensor(first, last, out, std::move(proj));
    return out;
}
// [unpack_to_tensor] }}}

auto bind_spin(PyObject*) -> void;
// auto bind_spin(pybind11::module) -> void;

TCM_NAMESPACE_END

namespace std {
/// Specialisation of std::hash for SpinVectors to use in QuantumState
template <> struct hash<::TCM_NAMESPACE::SpinVector> {
    auto operator()(::TCM_NAMESPACE::SpinVector const& spin) const noexcept
        -> size_t
    {
        return spin.hash();
    }
};
} // namespace std
