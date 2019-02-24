#pragma once

#include <boost/pool/pool_alloc.hpp>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <flat_hash_map/bytell_hash_map.hpp>
#include <gsl/gsl-lite.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <random>
#include <string>

#include <immintrin.h>
#include <endian.h>

#if defined(__clang__)
#    define TCM_CLANG                                                          \
        (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#    define TCM_ASSUME(cond) __builtin_assume(cond)
#    define TCM_LIKELY(cond) __builtin_expect(!!(cond), 1)
#    define TCM_UNLIKELY(cond) __builtin_expect(!!(cond), 0)
#elif defined(__GNUC__)
#    define TCM_GCC                                                            \
        (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#    define TCM_ASSUME(cond)                                                   \
        do {                                                                   \
            if (!(cond)) __builtin_unreachable();                              \
        } while (false)
#    define TCM_LIKELY(cond) __builtin_expect(!!(cond), 1)
#    define TCM_UNLIKELY(cond) __builtin_expect(!!(cond), 0)
#elif defined(_MSV_VER)
#    define TCM_MSVC _MSV_VER
#    define TCM_ASSUME(cond)                                                   \
        do {                                                                   \
        } while (false)
#    define TCM_LIKELY(cond) (cond)
#    define TCM_UNLIKELY(cond) (cond)
#else
#    error "Unsupported compiler."
#endif

#if defined(WIN32) || defined(_WIN32)
#    define TCM_EXPORT __declspec(dllexport)
#    define TCM_NOINLINE __declspec(noinline)
#    define TCM_FORCEINLINE __forceinline inline
#    define TCM_NORETURN
#    define TCM_HOT
#else
#    define TCM_EXPORT __attribute__((visibility("default")))
#    define TCM_NOINLINE __attribute__((noinline))
#    define TCM_FORCEINLINE __attribute__((always_inline)) inline
#    define TCM_NORETURN __attribute__((noreturn))
#    define TCM_HOT __attribute__((hot))
#endif

#if defined(NDEBUG)
#    define TCM_CONSTEXPR constexpr
#    define TCM_NOEXCEPT noexcept
#else
#    define TCM_CONSTEXPR
#    define TCM_NOEXCEPT
#endif

#define TCM_NAMESPACE tcm
#define TCM_NAMESPACE_BEGIN namespace tcm {
#define TCM_NAMESPACE_END } // namespace tcm

#if !defined(NDEBUG)
#    if defined(__cplusplus)
#        include <cassert>
#    else
#        include <assert.h>
#    endif
#    define TCM_ASSERT(cond, msg) assert((cond) && (msg)) /* NOLINT */
#else
#    define TCM_ASSERT(cond, msg)
#endif

#if defined(TCM_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-W"
#pragma GCC diagnostic warning "-Wall"
#pragma GCC diagnostic warning "-Wextra"
#pragma GCC diagnostic warning "-Wcast-align"
#pragma GCC diagnostic warning "-Wcast-qual"
#pragma GCC diagnostic warning "-Wctor-dtor-privacy"
#pragma GCC diagnostic warning "-Wdisabled-optimization"
#pragma GCC diagnostic warning "-Wformat=2"
#pragma GCC diagnostic warning "-Winit-self"
#pragma GCC diagnostic warning "-Wlogical-op"
#pragma GCC diagnostic warning "-Wmissing-declarations"
#pragma GCC diagnostic warning "-Wmissing-include-dirs"
#pragma GCC diagnostic warning "-Wnoexcept"
#pragma GCC diagnostic warning "-Wold-style-cast"
#pragma GCC diagnostic warning "-Woverloaded-virtual"
#pragma GCC diagnostic warning "-Wredundant-decls"
#pragma GCC diagnostic warning "-Wshadow"
#pragma GCC diagnostic warning "-Wsign-conversion"
#pragma GCC diagnostic warning "-Wsign-promo"
#pragma GCC diagnostic warning "-Wstrict-null-sentinel"
#pragma GCC diagnostic warning "-Wstrict-overflow=5"
#pragma GCC diagnostic warning "-Wswitch-default"
#pragma GCC diagnostic warning "-Wundef"
#pragma GCC diagnostic warning "-Wunused"
#endif

TCM_NAMESPACE_BEGIN

using std::size_t;
using std::uint16_t;
using std::uint64_t;

using real_type    = double;
using complex_type = std::complex<real_type>;

template <class T>
using optional = torch::optional<T>;

enum class Spin : unsigned char {
    down = 0x00,
    up   = 0x01,
};

namespace detail {
struct UnsafeTag {};
constexpr UnsafeTag unsafe_tag;

template <torch::ScalarType ScalarType>
using dtype_tag_t = std::integral_constant<torch::ScalarType, ScalarType>;


// Horizontally adds elements of a float4 vector.
//
// Solution taken from https://stackoverflow.com/a/35270026
TCM_FORCEINLINE static auto hadd(__m128 const v) noexcept -> float
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Horizontally adds elements of a float8 vector.
TCM_FORCEINLINE static auto hadd(__m256 const v) noexcept -> float
{
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow         = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hadd(vlow);
}
} // namespace detail

namespace detail {
auto make_what_message(char const* file, size_t line, char const* func,
                       std::string const& description) -> std::string;

auto spin_configuration_to_string(gsl::span<float const> spin) -> std::string;
} // namespace detail

#define TCM_ERROR(ExceptionType, ...)                                          \
    throw ExceptionType                                                        \
    {                                                                          \
        ::TCM_NAMESPACE::detail::make_what_message(                            \
            __FILE__, static_cast<size_t>(__LINE__), BOOST_CURRENT_FUNCTION,   \
            __VA_ARGS__)                                                       \
    }

#define TCM_CHECK(condition, ExceptionType, ...)                               \
    if (TCM_UNLIKELY(!(condition))) { TCM_ERROR(ExceptionType, __VA_ARGS__); } \
    do {                                                                       \
    } while (false)

// TCM_CHECK_DIM implementation details {{{
namespace detail {
constexpr auto is_dim_okay(int64_t const dimension, int64_t const expected)
    -> bool
{
    return dimension == expected;
}

constexpr auto is_dim_okay(int64_t const dimension, int64_t const expected_1,
                           int64_t const expected_2) -> bool
{
    return dimension == expected_1 || dimension == expected_2;
}

inline auto make_wrong_dim_msg(int64_t const dimension, int64_t const expected)
    -> std::string
{
    return fmt::format("wrong dimension {}; expected {}", dimension, expected);
}

inline auto make_wrong_dim_msg(int64_t const dimension,
                               int64_t const expected_1,
                               int64_t const expected_2) -> std::string
{
    return fmt::format("wrong dimension {}; expected either {} or {}",
                       dimension, expected_1, expected_2);
}
} // namespace detail
// }}}

#define TCM_CHECK_DIM(dimension, ...)                                          \
    TCM_CHECK(                                                                 \
        ::TCM_NAMESPACE::detail::is_dim_okay(dimension, __VA_ARGS__),          \
        std::domain_error,                                                     \
        ::TCM_NAMESPACE::detail::make_wrong_dim_msg(dimension, __VA_ARGS__))

// TCM_CHECK_SHAPE implementation details {{{
namespace detail {
constexpr auto is_shape_okay(int64_t const shape, int64_t const expected)
    -> bool
{
    return shape == expected;
}

constexpr auto is_shape_okay(std::tuple<int64_t, int64_t> const& shape,
                             std::tuple<int64_t, int64_t> const& expected)
    -> bool
{
    return shape == expected;
}

inline auto make_wrong_shape_msg(int64_t const shape, int64_t const expected)
    -> std::string
{
    return fmt::format("wrong shape [{}]; expected [{}]", shape, expected);
}

inline auto make_wrong_shape_msg(std::tuple<int64_t, int64_t> const& shape,
                                 std::tuple<int64_t, int64_t> const& expected)
    -> std::string
{
    return fmt::format("wrong shape [{}, {}]; expected [{}, {}]",
                       std::get<0>(shape), std::get<1>(shape),
                       std::get<0>(expected), std::get<1>(expected));
}
} // namespace detail
// }}}

#define TCM_CHECK_SHAPE(shape, expected)                                       \
    TCM_CHECK(::TCM_NAMESPACE::detail::is_shape_okay(shape, expected),         \
              std::domain_error,                                               \
              ::TCM_NAMESPACE::detail::make_wrong_shape_msg(shape, expected))

// [Errors] {{{
namespace detail {
TCM_NORETURN TCM_NOINLINE auto error_wrong_dim(char const* function,
                                               int64_t dim, int64_t expected)
    -> void;

TCM_NORETURN TCM_NOINLINE auto error_wrong_dim(char const* function,
                                               int64_t dim, int64_t expected1,
                                               int64_t expected2) -> void;

TCM_NORETURN TCM_NOINLINE auto error_wrong_shape(char const* function,
                                                 int64_t     shape,
                                                 int64_t     expected) -> void;

TCM_NORETURN TCM_NOINLINE auto
             error_wrong_shape(char const*                         function,
                               std::tuple<int64_t, int64_t> const& shape,
                               std::tuple<int64_t, int64_t> const& expected) -> void;

TCM_NORETURN TCM_NOINLINE auto error_not_contiguous(char const* function)
    -> void;

TCM_NORETURN TCM_NOINLINE auto error_wrong_type(char const*       function,
                                                torch::ScalarType type,
                                                torch::ScalarType expected)
    -> void;

TCM_NORETURN TCM_NOINLINE auto error_index_out_of_bounds(char const* function,
                                                         size_t i, size_t max)
    -> void;

TCM_NORETURN TCM_NOINLINE auto error_float_not_isfinite(char const* function,
                                                        float       x) -> void;

TCM_NORETURN TCM_NOINLINE auto error_float_not_isfinite(char const* function,
                                                        std::complex<float> x)
    -> void;

TCM_NORETURN TCM_NOINLINE auto error_float_not_isfinite(char const* function,
                                                        std::complex<double> x)
    -> void;
} // namespace detail
// [errors] }}}


// [SpinVector] {{{
class SpinVector {

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#endif
    union {
        struct {
            std::uint16_t spin[7];
            std::uint16_t size;
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
    constexpr SpinVector() noexcept : _data{} {}

    constexpr SpinVector(SpinVector const&) noexcept = default;
    constexpr SpinVector(SpinVector&&) noexcept      = default;
    constexpr SpinVector& operator=(SpinVector const&) noexcept = default;
    constexpr SpinVector& operator=(SpinVector&&) noexcept = default;

    SpinVector(float const* data, size_t n);
    SpinVector(float const* data, size_t n, detail::UnsafeTag);
    SpinVector(std::initializer_list<float> spins);

    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    SpinVector(pybind11::array_t<float, ExtraFlags> const& spins);
    SpinVector(pybind11::str str);

    SpinVector(torch::Tensor const& spins);
    SpinVector(torch::TensorAccessor<float, 1> accessor);

    inline constexpr auto        size() const noexcept -> unsigned;
    inline static constexpr auto max_size() noexcept -> unsigned;

    inline auto magnetisation() const noexcept -> int;

    inline constexpr auto operator[](unsigned const i) const
        & TCM_NOEXCEPT -> Spin;
    inline constexpr auto operator[](unsigned const i) && TCM_NOEXCEPT -> Spin;
    inline constexpr auto operator[](unsigned const i)
        & TCM_NOEXCEPT -> SpinReference;

    inline /*constexpr*/ auto at(unsigned const i) const& -> Spin;
    inline /*constexpr*/ auto at(unsigned const i) && -> Spin;
    inline /*constexpr*/ auto at(unsigned const i) & -> SpinReference;

    inline constexpr auto flip(unsigned i) TCM_NOEXCEPT -> void;

    inline constexpr auto
    flipped(std::initializer_list<unsigned> is) const TCM_NOEXCEPT
        -> SpinVector;

    inline auto operator==(SpinVector const& other) const TCM_NOEXCEPT -> bool;
    inline auto operator!=(SpinVector const& other) const TCM_NOEXCEPT -> bool;

    inline auto operator<(SpinVector const& other) const TCM_NOEXCEPT -> bool;

    inline auto     hash() const noexcept -> size_t;
    inline explicit operator uint64_t() const;
    inline explicit operator std::string() const;

    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    auto numpy(pybind11::array_t<float, ExtraFlags> out) const
        -> pybind11::array_t<float, ExtraFlags>;

    auto numpy() const -> pybind11::array_t<float, pybind11::array::c_style>;

    inline auto copy_to(float* buffer, size_t n) const TCM_NOEXCEPT -> void;

    auto ska_key() const noexcept -> int64_t { return _data.as_ints[0]; }

  private:
    // [SpinVector.private junk] {{{
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
                            detail::UnsafeTag) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(range.size() <= max_size(), "Spin chain too long");
        TCM_ASSERT(is_valid_spin(range), "Invalid spin configuration");
        static_cast<void>(range);
    }

    TCM_FORCEINLINE static auto load_u16_short(float const* data,
                                               size_t const n) TCM_NOEXCEPT
        -> uint16_t
    {
        TCM_ASSERT(n < 16, "Range too long");
        TCM_ASSERT(is_valid_spin({data, n}), "Invalid spin value");
        auto result = uint16_t{0};
        for (auto i = size_t{0}; i < n; ++i) {
            set_bit(result, i, data[i] == 1.0f ? Spin::up : Spin::down);
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

    auto copy_from(float const* data, size_t const n) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(n == 0 || data != nullptr, "Invalid range");
        _data.as_ints     = _mm_set1_epi32(0);
        _data.size        = static_cast<std::uint16_t>(n);
        auto const chunks = n / 16;
        auto const rest   = n % 16;
        for (auto i = size_t{0}; i < chunks; ++i, data += 16) {
            _data.spin[i] =
                load_u16(_mm256_loadu_ps(data), _mm256_loadu_ps(data + 8));
        }
        if (rest != 0) {
            if (n >= 16) {
                data -= (16u - rest);
                _data.spin[chunks] = static_cast<uint16_t>(
                    load_u16(_mm256_loadu_ps(data), _mm256_loadu_ps(data + 8))
                    << (16u - rest));
            }
            else {
                _data.spin[chunks] = load_u16_short(data, rest);
            }
        }
    }

    auto is_valid() const TCM_NOEXCEPT -> bool
    {
        for (auto i = size(); i < max_size(); ++i) {
            if (unsafe_at(i) != Spin::down) { return false; }
        }
        return true;
    }

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
inline constexpr auto SpinVector::size() const noexcept -> unsigned
{
    return _data.size;
}

inline constexpr auto SpinVector::max_size() noexcept -> unsigned
{
    return 8 * sizeof(_data.spin);
}

inline constexpr auto SpinVector::
                      operator[](unsigned const i) const& TCM_NOEXCEPT -> Spin
{
    TCM_ASSERT(i < size(), "Index out of bounds.");
    return unsafe_at(i);
}

inline constexpr auto SpinVector::operator[](unsigned const i)
    && TCM_NOEXCEPT -> Spin
{
    TCM_ASSERT(i < size(), "Index out of bounds.");
    return unsafe_at(i);
}

inline constexpr auto SpinVector::operator[](unsigned const i)
    & TCM_NOEXCEPT -> SpinReference
{
    TCM_ASSERT(i < size(), "Index out of bounds.");
    return unsafe_at(i);
}

inline /*constexpr*/ auto SpinVector::at(unsigned const i) const& -> Spin
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of bounds {}; expected <={}", i, size()));
    return unsafe_at(i);
}

inline /*constexpr*/ auto SpinVector::at(unsigned const i) && -> Spin
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of bounds {}; expected <={}", i, size()));
    return unsafe_at(i);
}

inline /*constexpr*/ auto SpinVector::at(unsigned const i) & -> SpinReference
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of bounds {}; expected <={}", i, size()));
    return unsafe_at(i);
}

inline constexpr auto SpinVector::flip(unsigned const i) TCM_NOEXCEPT -> void
{
    TCM_ASSERT(i < size(), "Index out of bounds.");
    auto const chunk = i / 16u;
    auto const rest  = i % 16u;
    flip_bit(_data.spin[chunk], rest);
}

inline constexpr auto
SpinVector::flipped(std::initializer_list<unsigned> is) const TCM_NOEXCEPT
    -> SpinVector
{
    SpinVector temp{*this};
    for (auto const i : is) {
        temp.flip(i);
    }
    return temp;
}

inline auto SpinVector::magnetisation() const noexcept -> int
{
    static_assert(sizeof(unsigned long) == sizeof(uint64_t),
                  "Oops! Please, submit a bug report.");
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
    return _data.as_ints[0] < other._data.as_ints[0];
}

inline auto SpinVector::hash() const noexcept -> size_t
{
    static_assert(sizeof(_data.as_ints[0]) == sizeof(size_t), "");

    auto const hash_uint64 = [](uint64_t x) noexcept->uint64_t
    {
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB;
        x = x ^ (x >> 31);
        return x;
    };

#if 0
    return hash_uint64(static_cast<uint64_t>(_data.as_ints[0]));
#else
    auto const hash_combine = [hash_uint64](uint64_t seed,
                                            uint64_t x) noexcept->uint64_t
    {
        seed ^=
            hash_uint64(x) + uint64_t{0x9E3779B9} + (seed << 6) + (seed >> 2);
        return seed;
    };

    return hash_combine(hash_uint64(static_cast<uint64_t>(_data.as_ints[0])),
                        hash_uint64(static_cast<uint64_t>(_data.as_ints[1])));
#endif
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

template <int ExtraFlags, class>
auto SpinVector::numpy(pybind11::array_t<float, ExtraFlags> out) const
    -> pybind11::array_t<float, ExtraFlags>
{
    TCM_CHECK_DIM(out.ndim(), 1);
    TCM_CHECK_SHAPE(out.shape(0), size());

    auto const spin2float = [](Spin const s) noexcept->float
    {
        return s == Spin::up ? 1.0f : -1.0f;
    };

    auto access = out.template mutable_unchecked<1>();
    for (auto i = 0u; i < size(); ++i) {
        access(i) = spin2float((*this)[i]);
    }
    return std::move(out);
}

auto SpinVector::numpy() const
    -> pybind11::array_t<float, pybind11::array::c_style>
{
    return numpy(pybind11::array_t<float, pybind11::array::c_style>{size()});
}

inline auto SpinVector::copy_to(float* const buffer,
                                size_t const n) const TCM_NOEXCEPT -> void
{
    TCM_ASSERT(n == size(), "Wrong buffer size");
    auto spin2float = [](Spin const s) noexcept->float
    {
        return s == Spin::up ? 1.0f : -1.0f;
    };

    for (auto i = 0u; i < size(); ++i) {
        buffer[i] = spin2float((*this)[i]);
    }
}

inline SpinVector::SpinVector(float const* const data, size_t const n,
                              detail::UnsafeTag /*unused*/)
{
    check_range({data, n}, detail::unsafe_tag);
    copy_from(data, n);
}

inline SpinVector::SpinVector(std::initializer_list<float> spins)
    : SpinVector{spins.begin(), spins.size()}
{}

template <int ExtraFlags, class = std::enable_if_t<
                              ExtraFlags & pybind11::array::c_style
                              || ExtraFlags & pybind11::array::f_style> /**/>
TCM_NOINLINE
SpinVector::SpinVector(pybind11::array_t<float, ExtraFlags> const& spins)
    : SpinVector{spins.template unchecked<1>().data(0),
                 static_cast<size_t>(spins.shape(0))}
{
    TCM_CHECK_DIM(spins.ndim(), 1);
    auto const* data = spins.data();
    auto const  size = spins.shape(0);
    TCM_ASSERT(size >= 0, "Bug in pybind11?");
    TCM_ASSERT(spins.strides(0) == static_cast<int64_t>(sizeof(float)),
               "Bug in pybind11?");
    check_range({data, static_cast<size_t>(size)});
    copy_from(data, static_cast<size_t>(size));
}

inline SpinVector::SpinVector(torch::TensorAccessor<float, 1> const accessor)
{
    if (TCM_UNLIKELY(accessor.stride(0) != 1))
        detail::error_not_contiguous(
            "SpinVector(torch::TensorAccessor<float, 1>)");
    auto const* data = accessor.data();
    auto const  size = static_cast<size_t>(accessor.size(0));
    check_range({data, size});
    copy_from(data, size);
}

SpinVector::SpinVector(float const* const data, size_t const n)
{
    check_range({data, n});
    copy_from(data, n);
}

SpinVector::SpinVector(torch::Tensor const& spins)
{
    static constexpr auto const* function =
        "SpinVector(torch::Tensor const& input)";
    TCM_CHECK_DIM(spins.dim(), 1);
    if (TCM_UNLIKELY(!spins.is_contiguous()))
        detail::error_not_contiguous(function);
    if (TCM_UNLIKELY(spins.type().scalarType() != torch::kFloat32))
        detail::error_wrong_type(function, spins.type().scalarType(),
                                 torch::kFloat32);

    auto const* data = spins.data<float>();
    auto const  size = static_cast<size_t>(spins.size(0));
    check_range({data, size});
    copy_from(data, size);
}


TCM_NOINLINE SpinVector::SpinVector(pybind11::str str)
{
    // PyUnicode_Check macro uses old style casts
#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
    // Borrowed from pybind11/pytypes.h
    pybind11::object temp = str;
    if (PyUnicode_Check(str.ptr())) {
        temp = pybind11::reinterpret_steal<pybind11::object>(
            PyUnicode_AsUTF8String(str.ptr()));
        if (!temp)
            throw std::runtime_error{
                "Unable to extract string contents! (encoding issue)"};
    }
    char*             buffer;
    pybind11::ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length)) {
        throw std::runtime_error{
            "Unable to extract string contents! (invalid type)"};
    }
    if (length > static_cast<pybind11::ssize_t>(max_size())) {
        throw std::overflow_error{
            "Spin chain too long: " + std::to_string(length)
            + "; expected <=" + std::to_string(max_size())};
    }

    _data.as_ints = _mm_set1_epi32(0);
    _data.size    = static_cast<std::uint16_t>(length);
    for (auto i = 0u; i < _data.size; ++i) {
        auto const s = buffer[i];
        if (s != '0' && s != '1') {
            throw std::domain_error{"Invalid spin: '" + std::string{s}
                                    + "'; expected '0' or '1'"};
        }
        unsafe_at(i) = (s == '1') ? Spin::up : Spin::down;
    }
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#endif
}
// }}}

namespace detail {
struct SpinHasher {
    auto operator()(SpinVector const& x) const noexcept { return x.hash(); }
};
} // namespace detail

// Tensor creation routines {{{
namespace detail {
inline auto make_f32_tensor(size_t n) -> torch::Tensor
{
    TCM_ASSERT(n <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "Integer overflow");
    auto out = torch::empty(
        {static_cast<int64_t>(n)},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    TCM_ASSERT(out.is_contiguous(), "Why is it not contiguous?");
    return out;
}

inline auto make_f32_tensor(size_t n, size_t m) -> torch::Tensor
{
    TCM_ASSERT(n <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "Integer overflow");
    TCM_ASSERT(m <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "Integer overflow");
    auto out = torch::empty(
        {static_cast<int64_t>(n), static_cast<int64_t>(m)},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    TCM_ASSERT(out.is_contiguous(), "Why is it not contiguous?");
    TCM_ASSERT(out.stride(0) == static_cast<int64_t>(m),
               "Oh no! We didn't account for padding...");
    return out;
}
} // namespace detail
// }}}

namespace detail {
#if 0
template <class Iterator,
          class = std::enable_if_t<std::is_convertible<
              typename std::iterator_traits<Iterator>::value_type::first_type,
              SpinVector>::value> /**/>
auto keys_to_tensor(Iterator begin, size_t const size)
    -> std::tuple<torch::Tensor, Iterator>
{
    if (size == 0) { return {make_f32_tensor(0), begin}; }
    // We assume that all chains in the range [begin, begin + size) have the
    // same size
    auto const number_spins = begin->first.size();
    auto       out          = make_f32_tensor(size, number_spins);
    auto const ldim         = out.stride(0);
    TCM_ASSERT(ldim >= static_cast<int64_t>(number_spins), "Huh?!");
    TCM_ASSERT(out.stride(1) == 1, "The tensor must be contiguous.");
    auto* data = out.template data<float>();
    for (auto i = size_t{0}; i < size; ++i, ++begin, data += ldim) {
        begin->first.copy_to(data, number_spins);
    }
    return {std::move(out), begin};
}

template <class Map, class = std::enable_if_t<std::is_same<
                         typename Map::key_type, SpinVector>::value> /**/>
auto keys_to_tensor(Map const& map) -> torch::Tensor
{
#if 1
    return std::get<0>(keys_to_tensor(map.begin(), map.size()));
#else
    if (map.empty()) {
        return torch::empty(
            {0},
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    }
    auto       begin        = map.begin();
    auto const number_spins = begin->first.size();
    auto       out          = torch::empty(
        {static_cast<int64_t>(map.size()), static_cast<int64_t>(number_spins)},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    auto const ldim = out.stride(0);
    TCM_ASSERT(ldim == static_cast<int64_t>(number_spins), "");
    TCM_ASSERT(out.stride(1) == 1, "");
    auto* data = out.template data<float>();
    for (auto i = size_t{0}; i < map.size(); ++i, ++begin, data += ldim) {
        begin->first.copy_to(data, number_spins);
    }
    return out;
#endif
}

struct RealPartAsFloatFn {
    template <class T>
    TCM_FORCEINLINE constexpr auto operator()(T const& x) const noexcept -> float
    {
        return static_cast<float>(x);
    }

    template <class T>
    TCM_FORCEINLINE constexpr auto operator()(std::complex<T> const& x) const
        noexcept -> float
    {
        return static_cast<float>(x.real());
    }
};

template <class Map, class Projection,
          class = std::enable_if_t<std::is_same<typename Map::mapped_type,
                                                complex_type>::value> /**/>
auto values_to_tensor(Map const& map, Projection&& proj = RealPartAsFloatFn{})
    -> torch::Tensor
{
    TCM_ASSERT(map.size()
                   <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "Integer overflow");
    auto out = make_f32_tensor(map.size());
    if (map.empty()) { return out; }
    auto begin = map.begin();
    auto data  = out.template accessor<float, 1>();
    for (auto i = int64_t{0}; i < static_cast<int64_t>(map.size());
         ++i, ++begin) {
        data[i] = static_cast<float>(begin->second.real());
    }
    return out;
}

struct IdentityFn {
    template <class T>
    TCM_FORCEINLINE constexpr decltype(auto) operator()(T&& x) const noexcept
    {
        return std::forward<T>(x);
    }
};

template <class ForwardIterator, class Projection = IdentityFn>
auto values_from_tensor(torch::Tensor const& values, ForwardIterator begin,
                        ForwardIterator end, Projection&& proj = IdentityFn{})
    -> void
{
    auto const dim  = values.dim();
    auto const size = values.size(0);
    TCM_CHECK_DIM(dim, 2);
    TCM_CHECK_SHAPE({size, values.size(1)}, {size, 1});
    auto const accessor = values.packed_accessor<float, 2>();
    for (auto i = int64_t{0}; i < size; ++i, ++begin) {
        TCM_ASSERT(begin != end, "Incompatible lengths");
        begin->second = proj(accessor[i][0]);
    }
}
#endif

#if 0
template <class Map, class Projection = IdentityFn>
auto values_from_tensor(torch::Tensor const& values, Map& map) -> void
{
    auto const size = static_cast<int64_t>(map.size());
    auto const dim  = values.dim();
    switch (dim) {
    case 1:
        if (values.size(0) != size) {
            std::ostringstream msg;
            msg << "`values` tensor has wrong shape: [" << values.size(0)
                << "]; expected [" << size << "]";
            throw std::domain_error{msg.str()};
        }
        break;
    case 2:
        if (values.size(0) != size || values.size(1) != 1) {
            std::ostringstream msg;
            msg << "`values` tensor has wrong shape: [" << values.size(0)
                << ", " << values.size(1) << "]; expected [" << size << ", 1]";
            throw std::domain_error{msg.str()};
        }
        break;
    default:
        throw std::domain_error{"Tensor has wrong dimension: "
                                + std::to_string(values.dim())
                                + "; expected 1 or 2"};
    } // end switch

    auto begin = map.begin();
    auto x        = dim == 1 ? values : values.select(1, 0);
    auto accessor = x.accessor<float, 1>();
    TCM_ASSERT(accessor.size(0) == size, "");
    for (auto i = int64_t{0}; i < size; ++i, ++begin) {
        begin->second = accessor[i];
    }
    // pybind11::print("values_from_tensor done");
}
#endif

template <class RandomAccessIterator>
auto unpack_to_tensor(RandomAccessIterator begin, RandomAccessIterator end,
                      torch::Tensor output)
{
    auto const size = end - begin;
    TCM_ASSERT(size > 0, "Input range must not be empty");
    auto const number_spins = begin->size();
    TCM_ASSERT(std::all_of(
                   begin, end,
                   [number_spins](auto const& x) { return x.size() == number_spins; }),
               "Input range contains variable size spin chains");
    TCM_ASSERT(output.dim() == 2, "Invalid dimension");
    TCM_ASSERT(size == output.size(0), "Sizes don't match");
    TCM_ASSERT(static_cast<int64_t>(number_spins) == output.size(1),
               "Sizes don't match");

    auto*      data = output.data<float>();
    auto const ldim = output.stride(0);
    for (auto i = int64_t{0}; i < size; ++i, ++begin, data += ldim) {
        begin->copy_to(data, number_spins);
    }
}

} // namespace detail

// [QuantumState] {{{
class QuantumState
    : public ska::bytell_hash_map<SpinVector, complex_type,
                                  detail::SpinHasher> {
  private:
    using base =
        ska::bytell_hash_map<SpinVector, complex_type, detail::SpinHasher>;
    using base::value_type;

    static_assert(alignof(base::value_type) == 16, "");
    static_assert(sizeof(base::value_type) == 32, "");

  public:
    using base::base;

    QuantumState(QuantumState const&) = default;
    QuantumState(QuantumState&&)      = default;
    QuantumState& operator=(QuantumState const&) = delete;
    QuantumState& operator=(QuantumState&&) = delete;

    TCM_FORCEINLINE TCM_HOT auto
                    operator+=(std::pair<complex_type, SpinVector> const& value)
        -> QuantumState&
    {
        TCM_ASSERT(std::isfinite(value.first.real())
                       && std::isfinite(value.first.imag()),
                   "Invalid coefficient");
        auto& c = (*this)[value.second];
        TCM_ASSERT(std::isfinite(c.real()) && std::isfinite(c.imag()),
                   "Invalid state");
        c += value.first;
        TCM_ASSERT(std::isfinite(c.real()) && std::isfinite(c.imag()),
                   "Invalid state");
        return *this;
    }

    friend auto swap(QuantumState& x, QuantumState& y) -> void
    {
        using std::swap;
        static_cast<base&>(x).swap(static_cast<base&>(y));
    }
};
// [QuantumState] }}}

// [Heisenberg] {{{
/// \brief Represents the Heisenberg Hamiltonian.
class Heisenberg {
  public:
    using edge_type = std::pair<unsigned, unsigned>;

  private:
    std::vector<edge_type> _edges; ///< Graph edges
    real_type _coupling;           ///< Coupling J. It should usually be taken
                                   ///< ≈1/#edges to ensure that the operator
                                   ///< norm of H is less than 1.
    unsigned _max_index; ///< The greatest site index present in `_edges`.
                         ///< It is used to detect errors when one tries to
                         ///< apply the hamiltonian to a spin configuration
                         ///< which is too short.

  public:
    /// Constructs a hamiltonian given graph edges and the coupling.
    ///
    /// \precondition coupling is normal, i.e. neither zero, infinite or NaN.
    Heisenberg(std::vector<edge_type> edges, real_type coupling);

    /// Copy and Move constructors/assignments
    Heisenberg(Heisenberg const&)     = default;
    Heisenberg(Heisenberg&&) noexcept = default;
    Heisenberg& operator=(Heisenberg const&) = default;
    Heisenberg& operator=(Heisenberg&&) noexcept = default;

    /// Returns the number of edges in the graph
    /*constexpr*/ auto size() const noexcept -> size_t { return _edges.size(); }

    /// Returns the coupling.
    constexpr auto coupling() const noexcept -> real_type { return _coupling; }
    /// Updates the coupling.
    inline auto coupling(real_type coupling) -> void;

    /// Returns the greatest index encountered in `_edges`.
    ///
    /// \precondition `size() != 0`
    /*constexpr*/ auto max_index() const noexcept -> size_t
    {
        TCM_ASSERT(!_edges.empty(), "_max_index is not defined");
        return _max_index;
    }

    /// Returns a *reference* to graph edges.
    constexpr auto const& edges() const noexcept { return _edges; }

    /// Performs `|ψ⟩ += c * H|σ⟩`.
    ///
    /// \param coeff Coefficient `c`
    /// \param spin  Spin configuration `|σ⟩`
    /// \param psi   State `|ψ⟩`
    ///
    /// \precondition `coeff` is finite, i.e.
    ///               `isfinite(coeff.real()) && isfinite(coeff.imag())`.
    /// \preconfition When `size() != 0`, `max_index() < spin.size()`.
    TCM_FORCEINLINE TCM_HOT auto operator()(complex_type const coeff,
                                            SpinVector const   spin,
                                            QuantumState& psi) const -> void
    {
        TCM_ASSERT(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
                   "Coefficient `coeff` must be a finite complex number");
        TCM_ASSERT(_edges.empty() || max_index() < spin.size(),
                   "Index out of bounds: `spin` is too short");
        auto c = complex_type{0, 0};
        for (auto const& edge : edges()) {
            // Heisenberg hamiltonian works more or less like this:
            //
            //     K|↑↑⟩ = J|↑↑⟩
            //     K|↓↓⟩ = J|↓↓⟩
            //     K|↑↓⟩ = -J|↑↓⟩ + 2J|↓↑⟩
            //     K|↓↑⟩ = -J|↓↑⟩ + 2J|↑↓⟩
            //
            // where K is the "kernel". We want to perform
            // |ψ⟩ += c * K|σᵢσⱼ⟩ for each edge (i, j).
            //
            auto const aligned = spin[edge.first] == spin[edge.second];
            // sign == 1.0 when aligned == true and sign == -1.0 when aligned == false
            auto const sign    = static_cast<real_type>(-1 + 2 * aligned);
            c += sign * coeff * coupling();
            if (!aligned) {
                psi += {real_type{2} * coeff * coupling(),
                        spin.flipped({edge.first, edge.second})};
            }
        }
        psi += {c, spin};
    }

  private:
    /// Finds the largest index used in `_edges`.
    ///
    /// \precondition Range must not be empty.
    template <class Iter, class = std::enable_if_t<std::is_same<
                              typename std::iterator_traits<Iter>::value_type,
                              edge_type>::value> /**/>
    auto find_max_index(Iter begin, Iter end) -> unsigned
    {
        TCM_ASSERT(begin != end, "Range is empty");
        // This implementation is quite inefficient, but it's not on the hot
        // bath, so who cares ;)
        auto max_index = std::max(begin->first, begin->second);
        for (; begin != end; ++begin) {
            max_index =
                std::max(max_index, std::max(begin->first, begin->second));
        }
        return max_index;
    }
};

Heisenberg::Heisenberg(std::vector<edge_type> edges, real_type const coupling)
    : _edges{std::move(edges)}
    , _coupling{coupling}
    , _max_index{std::numeric_limits<unsigned>::max()}
{
    TCM_CHECK(std::isnormal(coupling), std::invalid_argument,
              fmt::format("invalid coupling: {}; expected a normal (i.e. "
                          "neither zero, subnormal, infinite or NaN) float",
                          coupling));
    if (!_edges.empty()) {
        _max_index = find_max_index(std::begin(_edges), std::end(_edges));
    }
}

inline auto Heisenberg::coupling(real_type const coupling) -> void
{
    TCM_CHECK(std::isnormal(coupling), std::invalid_argument,
              fmt::format("invalid coupling: {}; expected a normal (i.e. "
                          "neither zero, subnormal, infinite or NaN) float",
                          coupling));
    _coupling = coupling;
}
// [Heisenberg] }}}

// [Polynomial] {{{
///
///
///
class Polynomial {

  public:
    /// `P[ε](H - A)` term.
    struct Term {
        complex_type        root; ///< A in the formula above
        optional<real_type> epsilon; ///< ε in the formula above
    };

  private:
    QuantumState            _current;
    QuantumState            _old;
    std::shared_ptr<Heisenberg const>
        _hamiltonian; ///< Hamiltonian which knows how to perform `|ψ⟩ += c * H|σ⟩`.
    std::vector<Term>       _terms; /// 
    std::vector<SpinVector> _basis;
    torch::Tensor           _coeffs;

  public:
    /// Constructs the polynomial given the hamiltonian and a list or terms.
    Polynomial(std::shared_ptr<Heisenberg const> hamiltonian,
               std::vector<Term>                 terms);

    Polynomial(Polynomial const&) = default;
    Polynomial(Polynomial &&) = default;
    Polynomial& operator=(Polynomial const&) = delete;
    Polynomial& operator=(Polynomial &&) = delete;

    inline auto degree() const noexcept -> size_t;
    inline auto size() const noexcept -> size_t;
    inline auto clear() noexcept -> void;

    // constexpr auto items() const & noexcept -> QuantumState const&;

    TCM_NOINLINE TCM_HOT auto operator()(complex_type const coeff,
                                         SpinVector const spin) -> Polynomial&;

    constexpr auto const& vectors() const noexcept { return _basis; }

    constexpr auto coefficients() const noexcept -> torch::Tensor const&
    {
        return _coeffs;
    }

  private:
    template <class Map>
    TCM_NOINLINE auto save_results(Map const&                        map,
                                   torch::optional<real_type> const& eps)
        -> void;

    template <class Map>
    static auto check_all_real(Map const& map) -> void
    {
        constexpr auto eps       = static_cast<real_type>(2e-3);
        auto           norm_full = real_type{0};
        auto           norm_real = real_type{0};
        for (auto const& item : map) {
            norm_full += std::norm(item.second);
            norm_real += std::norm(item.second.real());
        }
        if (norm_full >= (real_type{1} + eps) * norm_real) {
            throw std::runtime_error{
                "Polynomial contains complex coefficients: |P| >= (1 + eps) * "
                "|Re[P]|: |"
                + std::to_string(norm_full) + "| >= (1 + " + std::to_string(eps)
                + ") * |" + std::to_string(norm_real) + "|"};
        }
    }
};

Polynomial::Polynomial(std::shared_ptr<Heisenberg const> hamiltonian,
                       std::vector<Term>                 terms)
    : _current{}
    , _old{}
    , _hamiltonian{std::move(hamiltonian)}
    , _terms{std::move(terms)}
    , _basis{}
    , _coeffs{}
{
    if (_hamiltonian == nullptr) {
        throw std::invalid_argument{
            "Polynomial(shared_ptr<Heisenberg>, vector<pair<complex_type, "
            "optional<real_type>>>): hamiltonian must not be nullptr"};
    }
    auto const estimated_size =
        std::min(static_cast<size_t>(std::round(
                     std::pow(_hamiltonian->size() / 2, _terms.size()))),
                 size_t{4096});
    _old.reserve(estimated_size);
    _current.reserve(estimated_size);
}

inline auto Polynomial::degree() const noexcept -> size_t
{
    return _terms.size();
}

inline auto Polynomial::size() const noexcept -> size_t
{
    return _basis.size();
}

inline auto Polynomial::clear() noexcept -> void
{
    _current.clear();
    _old.clear();
    _basis.clear();
}

#if 0
constexpr auto Polynomial::items() const & noexcept -> QuantumState const&
{
    return _old;
}
#endif

template <class Map>
auto Polynomial::save_results(Map const& map, optional<real_type> const& eps)
    -> void
{
    // TODO(twesterhout): We might be seriously wasting memory here (i.e.
    // using ~5% of the allocated storage).
    auto const size = map.size();
    _basis.clear();
    _basis.reserve(size);
    if (!_coeffs.defined()) { _coeffs = detail::make_f32_tensor(size); }
    else if (_coeffs.size(0) != static_cast<int64_t>(size)) {
        _coeffs.resize_({static_cast<int64_t>(size)});
    }

    auto i        = int64_t{0};
    auto accessor = _coeffs.packed_accessor<float, 1>();
    if (eps.has_value()) {
        for (auto const& item : map) {
            if (std::abs(item.second.real()) >= *eps) {
                _basis.emplace_back(item.first);
                accessor[i++] = static_cast<float>(item.second.real());
            }
        }
        _coeffs.resize_(i);
    }
    else {
        for (auto const& item : map) {
            _basis.emplace_back(item.first);
            accessor[i++] = static_cast<float>(item.second.real());
        }
        TCM_ASSERT(i == static_cast<int64_t>(size), "");
    }
}

auto Polynomial::operator()(complex_type const coeff, SpinVector const spin)
    -> Polynomial&
{
    using std::swap;
    constexpr auto const* function =
        "Polynomial::operator()(complex_type, SpinVector)";
    if (TCM_UNLIKELY(!std::isfinite(coeff.real())
                     || !std::isfinite(coeff.imag())))
        detail::error_float_not_isfinite(function, coeff);
    if (TCM_UNLIKELY(_hamiltonian->max_index() >= spin.size()))
        detail::error_index_out_of_bounds(function, _hamiltonian->max_index(),
                                          spin.size());
    if (_terms.empty()) {
        _old.clear();
        _old.emplace(spin, coeff);
        save_results(_old, torch::nullopt);
        return *this;
    }

    // The zeroth iteration: goal is to perform `_old := coeff * (H - root)|spin⟩`
    {
        // `|_old⟩ := - coeff * root|spin⟩`
        _old.clear();
        _old.emplace(spin, -_terms[0].root * coeff);
        // `|_old⟩ += coeff * H|spin⟩`
        (*_hamiltonian)(coeff, spin, _old);
    }
    // Other iterations
    TCM_ASSERT(_current.empty(), "Bug!");
    for (auto i = size_t{1}; i < _terms.size(); ++i) {
        auto const  root    = _terms[i].root;
        auto const& epsilon = _terms[i - 1].epsilon;
        if (epsilon.has_value()) {
            // Performs `|_current⟩ := (H - root)P[epsilon]|_old⟩` in two steps:
            // 1) `|_current⟩ := - root * P[epsilon]|_old⟩`
            for (auto const& item : _old) {
                if (std::abs(item.second.real()) >= *epsilon) {
                    _current.emplace(item.first, -root * item.second);
                }
            }
            // 2) `|_current⟩ += H P[epsilon]|_old⟩`
            for (auto const& item : _old) {
                if (std::abs(item.second.real()) >= *epsilon) {
                    (*_hamiltonian)(item.second, item.first, _current);
                }
            }
        }
        else {
            // Performs `|_current⟩ := (H - root)|_old⟩` in two steps:
            // 1) `|_current⟩ := - root|_old⟩`
            for (auto const& item : _old) {
                _current.emplace(item.first, -root * item.second);
            }
            // 2) `|_current⟩ += H |_old⟩`
            for (auto const& item : _old) {
                (*_hamiltonian)(item.second, item.first, _current);
            }
        }
        // |_old⟩ := |_current⟩, but to not waste allocated memory, we use
        // `swap + clear` instead.
        swap(_old, _current);
        _current.clear();
    }
    // Final filtering, i.e. `P[epsilon] |_old⟩`
    save_results(_old, _terms.back().epsilon);
    return *this;
}
// [Polynomial] }}}

#if 0
class ToDo {

    // [CacheCell] {{{
    struct CacheCell {
      private:
        real_type _value;
        bool      _known;

      public:
        constexpr CacheCell() noexcept : _value{0}, _known{false} {}

        constexpr CacheCell(CacheCell const&) noexcept = default;
        constexpr CacheCell(CacheCell&&) noexcept      = default;
        constexpr CacheCell& operator=(CacheCell const&) noexcept = default;
        constexpr CacheCell& operator=(CacheCell&&) noexcept = default;

        constexpr auto operator=(real_type const value) TCM_NOEXCEPT
            -> CacheCell&
        {
            TCM_ASSERT(std::isfinite(value), "Invalid value");
            TCM_ASSERT(!has_value()
                           || std::abs(_value - value)
                                  < 0.001 * std::abs(value),
                       "Net must return the same value every time");
            _value = value;
            _known = true;
            return *this;
        }

        constexpr auto has_value() const noexcept -> bool { return _known; }

        constexpr auto value() const TCM_NOEXCEPT -> real_type
        {
            TCM_ASSERT(has_value(), "Value is not (yet) known");
            return _value;
        }
    };
    // [CacheCell] }}}

    static_assert(sizeof(CacheCell) <= sizeof(SpinVector), "");

    template <class T>
    using pool_allocator =
        boost::fast_pool_allocator<T, boost::default_user_allocator_new_delete,
                                   boost::details::pool::null_mutex>;

    using map_type = std::unordered_map<
        SpinVector, CacheCell, detail::SpinHasher, std::equal_to<SpinVector>,
        pool_allocator<std::pair<SpinVector const, CacheCell>>>;

    /// A task of computing ⟨σ|P[H]|ψ⟩ for one σ.
    class Task {

        // [Term] {{{
        struct Term {
          private:
            real_type        _coeff;
            CacheCell const* _cell;

          public:
            constexpr Term(real_type const  coeff,
                           CacheCell const& cell) TCM_NOEXCEPT
                : _coeff{coeff}
                , _cell{&cell}
            {
                TCM_ASSERT(std::isfinite(coeff), "Invalid coefficient");
            }

            constexpr Term(Term const&) noexcept = default;
            constexpr Term(Term&&) noexcept      = default;
            constexpr Term& operator=(Term const&) noexcept = default;
            constexpr Term& operator=(Term&&) noexcept = default;

            constexpr auto has_value() const noexcept -> bool
            {
                return _cell->has_value();
            }

            constexpr auto value() const TCM_NOEXCEPT -> real_type
            {
                TCM_ASSERT(has_value(), "Value is not yet known");
                return _coeff * _cell->value();
            }
        };
        // [Term] }}}

        static_assert(sizeof(Term) == 2 * sizeof(void*), "");

        using buffer_type = std::vector<Term>;

        buffer_type _buffer;
        real_type   _bias;

      public:
        Task() : _buffer{}, _bias{0} {}
        Task(Polynomial const& poly, map_type& todo) { reset(poly, todo); }

        template <class SearchFn>
        Task(Polynomial const& poly, map_type& todo, SearchFn&& search_fn)
        {
            reset(poly, todo, std::forward<SearchFn>(search_fn));
        }

        Task(Task const&)     = delete;
        Task(Task&&) noexcept = default;
        Task& operator=(Task const&) = delete;
        Task& operator=(Task&&) noexcept = default;

        auto reset(Polynomial const& poly, map_type& todo) -> void
        {
            struct DummySearchFn {
                constexpr auto operator()(SpinVector const& /*unused*/) const
                    noexcept -> real_type const*
                {
                    return nullptr;
                }
            };
            reset(poly, todo, DummySearchFn{});
        }

        template <class SearchFn>
        auto reset(Polynomial const& poly, map_type& todo, SearchFn&& search_fn)
            -> void
        {
            _bias = real_type{0};
            _buffer.clear();
            _buffer.reserve(poly.size());

            for (auto const& item : poly.items()) {
                auto const& spin  = item.first;
                auto const& coeff = item.second;
                auto const* p     = search_fn(spin);
                if (p != nullptr) {
                    auto const value = *p;
                    _bias += coeff.real() * value;
                }
                else {
                    auto const iterator = todo.emplace(spin, CacheCell{}).first;
                    _buffer.emplace_back(coeff.real(), iterator->second);
                }
            }
        }

        auto update() -> void
        {
            auto count = size_t{0};
            auto const pred  = [](auto const& x) { return x.has_value(); };
            auto const size = _buffer.size();
            for (auto const& item : _buffer) {
                if (item.has_value()) {
                    _bias += item.value();
                    ++count;
                }
            }
            auto last = std::remove_if(_buffer.begin(), _buffer.end(), pred);
            _buffer.erase(last, _buffer.end());
            TCM_ASSERT(size - _buffer.size() == count, "");
            TCM_ASSERT((std::none_of(_buffer.begin(), _buffer.end(), pred)), "");
            // auto const last  = _buffer.end();
            // auto       first = std::find_if(_buffer.begin(), last, pred);

            // if (first != last) {
            //     auto i = first;
            //     _bias += i->value();
            //     while (++i != last) {
            //         if (!pred(*i)) {
            //             *first = std::move(*i);
            //             ++first;
            //         }
            //         else {
            //             _bias += i->value();
            //         }
            //     }
            //     _buffer.erase(first, last);
            // }
        }

        auto get() const -> real_type
        {
            auto total = _bias;
            for (auto const& item : _buffer) {
                total += item.value();
            }
            TCM_ASSERT(std::isfinite(total), "Invalid value");
            return total;
        }
    };

  private:
    map_type          _todo;
    std::vector<Task> _tasks;
    size_t            _number_tasks;

  public:
    ToDo() : _todo{}, _tasks{}/*, _number_tasks{0}*/ {}
    ToDo(ToDo const&) = delete;
    ToDo(ToDo&&)      = delete;
    ToDo& operator=(ToDo const&) = delete;
    ToDo& operator=(ToDo&&) = delete;

    auto clear() -> void
    {
        _todo.clear();
        // _tasks.clear();
        _number_tasks = 0;
    }

    auto size() const noexcept -> size_t { return _todo.size(); }

    auto insert(Polynomial const& poly) -> void
    {
        struct DummySearchFn {
            constexpr auto operator()(SpinVector const& /*unused*/) const
                noexcept -> real_type const*
            {
                return nullptr;
            }
        };
        insert(poly, DummySearchFn{});
    }

    template <class SearchFn>
    auto insert(Polynomial const& poly, SearchFn&& search_fn) -> void
    {
        TCM_ASSERT(_number_tasks <= _tasks.size(), "Bug!");
        if (_number_tasks == _tasks.size()) {
            _tasks.emplace_back(poly, _todo, std::forward<SearchFn>(search_fn));
        }
        else {
            _tasks[_number_tasks].reset(poly, _todo,
                                        std::forward<SearchFn>(search_fn));
        }
        ++_number_tasks;
    }

  private:
    // auto values_from_tensor(torch::Tensor const& values,
    //                         map_type::iterator begin, map_type::iterator end)
    //     -> void
    // {
    //     auto const dim  = values.dim();
    //     if (TCM_UNLIKELY(dim != 2)) {
    //         throw std::domain_error{"Tensor has wrong dimension: "
    //                                 + std::to_string(dim)
    //                                 + "; expected 2"};
    //     }
    //     if (TCM_UNLIKELY(values.size(1) != 1)) {
    //         std::ostringstream msg;
    //         msg << "`values` tensor has wrong shape: [" << values.size(0)
    //             << ", " << values.size(1) << "]; expected [??, 1]";
    //         throw std::domain_error{msg.str()};
    //     }

    //     auto const size     = values.size(0);
    //     auto       accessor = values.accessor<float, 2>();
    //     auto       i        = int64_t{0};
    //     for (; i < size && begin != end; ++i, ++begin) {
    //         begin->second = accessor[i][0];
    //     }

    //     if (TCM_UNLIKELY(begin != end || i != size)) {
    //         std::ostringstream msg;
    //         msg << "`values` tensor has wrong shape: [" << values.size(0)
    //             << ", " << values.size(1) << "]";
    //         throw std::domain_error{msg.str()};
    //     }
    // }

  public:
    template <class Net>
    auto evaluate_batch(Net&& net, size_t batch_size) -> void
    {
        TCM_ASSERT(_todo.size() >= batch_size && batch_size > 0,
                   "Bug! This function should only be called when the todo "
                   "list contains at least one batch of data");

        auto const begin   = _todo.begin();
        auto       _result = detail::keys_to_tensor(begin, batch_size);
        auto       keys    = std::move(std::get<0>(_result));
        auto const end     = std::move(std::get<1>(_result));
        // auto       keys         = torch::empty(
        //     {static_cast<int64_t>(batch_size),
        //      static_cast<int64_t>(number_spins)},
        //     torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));

        // auto const ldim = keys.stride(0);
        // TCM_ASSERT(keys.stride(0) == number_spins, "");
        // TCM_ASSERT(keys.stride(1) == 1, "");
        // auto*      data = keys.template data<float>();
        // for (auto i = size_t{0}; i < batch_size; ++i, ++begin, data += ldim) {
        //     begin->first.copy_to(data, number_spins);
        //     TCM_ASSERT((begin->first == SpinVector{data, number_spins}), "");
        // }

        auto const values = net.forward(keys, begin, end);
        detail::values_from_tensor(values, begin, end);

        for (auto i = size_t{0}; i < _number_tasks; ++i) {
            _tasks[i].update();
        }

        _todo.erase(begin, end);
    }

    template <class Net>
    auto evaluate(Net&& net, size_t /*batch_size*/) -> torch::Tensor
    {
        if (!_todo.empty()) {
            auto const values = net.forward(detail::keys_to_tensor(_todo),
                                            _todo.begin(), _todo.end());
            detail::values_from_tensor(values, _todo.begin(), _todo.end());
        }

        auto out      = detail::make_f32_tensor(_tasks.size());
        auto accessor = out.accessor<float, 1>();
        for (auto i = size_t{0}; i < _number_tasks; ++i) {
            accessor[static_cast<int64_t>(i)] = _tasks[i].get();
        }
        return out;
    }
};
#endif

namespace detail {
inline auto load_forward_fn(std::string const& filename)
    -> std::function<auto(torch::Tensor const&)->torch::Tensor>
{
    auto m = torch::jit::load(filename);
    if (m == nullptr) {
        throw std::runtime_error{
            "Failed to load torch::jit::script::Module from '" + filename
            + "'"};
    }
    auto fn = 
        [module = std::move(m)](torch::Tensor const& input) -> torch::Tensor {
            return module->forward({input}).toTensor();
        };
    return fn;
}
} // namespace detail

#if 0
class Machine final : public std::enable_shared_from_this<Machine> {

    using real_type = float;

    struct CacheCell {
        real_type value;
    };

    using cache_type =
        ska::bytell_hash_map<SpinVector, CacheCell, detail::SpinHasher>;

    using forward_fn_type =
        std::function<auto(torch::Tensor const&)->torch::Tensor>;

  private:
    forward_fn_type _psi;
    cache_type      _cache;

    template <class T>
    auto forward_impl_1d(torch::Tensor const& input) -> torch::Tensor;

    template <class T>
    auto forward_impl_2d(torch::Tensor const& input) -> torch::Tensor;

  public:
    Machine(forward_fn_type net) : _psi{std::move(net)}, _cache{}
    {
        // auto x = torch::zeros({1, 10}, torch::TensorOptions()
        //                                    .dtype(torch::kFloat32)
        //                                    .requires_grad(false));
        // x[0][0] = 1;
        // x[0][1] = 1;
        // x[0][2] = -1;
        // x[0][3] = -1;
        // x[0][4] = -1;
        // x[0][5] = 1;
        // x[0][6] = -1;
        // x[0][7] = 1;
        // x[0][8] = 1;
        // x[0][9] = -1;
        // // auto x = torch::tensor(std::vector<float>{1, 1, -1, -1, -1, 1, -1, 1, 1, -1});
        // pybind11::print((*this)(x));
        // pybind11::print((*this)(x));
    }

    Machine(Machine const&) = delete;
    Machine(Machine&&)      = delete;
    Machine& operator=(Machine const&) = delete;
    Machine& operator=(Machine&&) = delete;

    constexpr auto const& cache() const noexcept { return _cache; }

    constexpr auto psi() const noexcept -> forward_fn_type const&
    {
        return _psi;
    }

    auto search_fn() const noexcept
    {
        struct SearchFn {
            auto operator()(SpinVector const& x) const noexcept
                -> real_type const*
            {
                auto const where = cache.find(x);
                if (where == cache.end()) return nullptr;
                return std::addressof(where->second.value);
            }

            cache_type const& cache;
        };

        return SearchFn{_cache};
    }

    auto operator()(torch::Tensor const& input, bool use_cache = true)
        -> torch::Tensor
    {
        return forward(input, use_cache);
    }

    template <class ForwardIterator>
    auto operator()(torch::Tensor const& input, ForwardIterator begin,
                    ForwardIterator end) -> torch::Tensor
    {
        return forward(input, begin, end);
    }

    auto forward(torch::Tensor const& input, bool use_cache = true)
        -> torch::Tensor;

    template <class ForwardIterator>
    auto forward(torch::Tensor const& input, ForwardIterator begin,
                 ForwardIterator end) -> torch::Tensor;
};

template <>
auto Machine::forward_impl_1d<float>(torch::Tensor const& input)
    -> torch::Tensor
{
    TCM_ASSERT(input.type().scalarType() == torch::kFloat32,
               "Bug! This function should only be called with Float32 Tensors");
    TCM_ASSERT(input.dim() == 1,
               "Bug! This function should only be called with 1D Tensors");

    auto const spin  = SpinVector{input};
    auto const where = _cache.find(spin);
    if (where != _cache.end()) {
        return torch::tensor(
            static_cast<float>(where->second.value),
            torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    }
    auto value = _psi(input);
    _cache.emplace(spin, CacheCell{value.template item<float>()});
    return value;
}

template <>
auto Machine::forward_impl_2d<float>(torch::Tensor const& input)
    -> torch::Tensor
{
    constexpr auto const* function =
        "Machine::forward_impl_2d<float>(torch::Tensor const&): the underlying "
        "machine behaves weirdly";
    TCM_ASSERT(input.type().scalarType() == torch::kFloat32,
               "Bug! This function should only be called with Float32 Tensors");
    TCM_ASSERT(input.dim() == 2,
               "Bug! This function should only be called with 2D Tensors");

    auto output = _psi(input);
    if (TCM_UNLIKELY(output.dim() != 2))
        detail::error_wrong_dim(function, output.dim(), 2);
    if (TCM_UNLIKELY(output.size(0) != input.size(0)))
        detail::error_wrong_shape(function, {output.size(0), output.size(1)},
                                  {input.size(0), 1});
    return output;
}

auto Machine::forward(torch::Tensor const& input, bool use_cache)
    -> torch::Tensor
{
    constexpr auto const* function =
        "Machine::forward(torch::Tensor const&, bool)";
    torch::NoGradGuard no_grad;
    if (!use_cache) { return _psi(input); }

    auto const scalar_type = input.type().scalarType();
    if (TCM_UNLIKELY(scalar_type != torch::kFloat32))
        detail::error_wrong_type(function, scalar_type, torch::kFloat32);

    auto const dim = input.dim();
    switch (dim) {
    case 1: return forward_impl_1d<float>(input);
    case 2: return forward_impl_2d<float>(input);
    default: detail::error_wrong_dim(function, dim, 1, 2);
    } // end switch
}

template <class ForwardIterator>
auto Machine::forward(torch::Tensor const& input, ForwardIterator begin,
                      ForwardIterator end) -> torch::Tensor
{
    constexpr auto const* function = "Machine::forward(torch::Tensor const&, "
                                     "ForwardIterator, ForwardIterator)";
    torch::NoGradGuard no_grad;

    auto const scalar_type = input.type().scalarType();
    if (TCM_UNLIKELY(scalar_type != torch::kFloat32))
        detail::error_wrong_type(function, scalar_type, torch::kFloat32);
    auto const dim = input.dim();
    if (TCM_UNLIKELY(dim != 2)) detail::error_wrong_dim(function, dim, 2);

    auto       output        = forward_impl_2d<float>(input);
    auto const accessor      = output.packed_accessor<float, 2>();
    auto       number_wasted = size_t{0};
    _cache.reserve(_cache.size() + static_cast<size_t>(accessor.size(0)));
    for (auto i = int64_t{0}; i < accessor.size(0); ++i, ++begin) {
        TCM_ASSERT(begin != end, "Bug! Incompatible lengths");
        auto const success =
            _cache.emplace(begin->first, CacheCell{accessor[i][0]}).second;
        number_wasted += static_cast<size_t>(!success);
    }
    TCM_ASSERT(number_wasted == 0, "");
    static_cast<void>(end);
    return output;
}
#endif

struct VarAccumulator {
    using value_type = real_type;

  private:
    size_t     _count;
    value_type _mean;
    value_type _M2;

  public:
    constexpr VarAccumulator() noexcept : _count{0}, _mean{}, _M2{} {}

    constexpr VarAccumulator(VarAccumulator const&) noexcept = default;
    constexpr VarAccumulator(VarAccumulator&&) noexcept      = default;
    constexpr VarAccumulator&
                              operator=(VarAccumulator const&) noexcept = default;
    constexpr VarAccumulator& operator=(VarAccumulator&&) noexcept = default;

    constexpr auto operator()(value_type const x) noexcept -> void
    {
        ++_count;
        auto const delta = x - _mean;
        _mean += delta / _count;
        _M2 += delta * (x - _mean);
    }

    constexpr auto count() const noexcept -> size_t { return _count; }

    auto mean() const -> value_type
    {
        TCM_CHECK(_count > 0, std::runtime_error,
                  fmt::format("mean of 0 samples is not defined"));
        return _mean;
    }

    auto variance() const -> value_type
    {
        TCM_CHECK(_count > 1, std::runtime_error,
                  fmt::format("sample variance of {} samples is not defined",
                              _count));
        return _M2 / (_count - 1);
    }
};

static_assert(std::is_trivially_copyable<VarAccumulator>::value, "");
static_assert(std::is_trivially_destructible<VarAccumulator>::value, "");

class PolynomialState {
    using FunctionT = std::function<auto(torch::Tensor const&)->torch::Tensor>;
  private:
    FunctionT      _psi;
    Polynomial     _poly;
    size_t         _batch_size;
    torch::Tensor  _input;
    VarAccumulator _poly_time;
    VarAccumulator _psi_time;

  public:
    PolynomialState(FunctionT psi, Polynomial poly, size_t const batch_size)
        : _psi{std::move(psi)}
        , _poly{std::move(poly)}
        , _batch_size{batch_size}
        , _input{}
        , _poly_time{}
        , _psi_time{}
    {}

    PolynomialState(PolynomialState const&) = delete;
    PolynomialState(PolynomialState&&) = delete;
    PolynomialState& operator=(PolynomialState const&) = delete;
    PolynomialState& operator=(PolynomialState&&) = delete;

    /// Returns the batch size used internally for forward propagation
    constexpr auto batch_size() const noexcept -> size_t { return _batch_size; }

    /// Updates the batch size used internally for forward propagation. Note
    /// that this function may allocate and is thus not marked `noexcept`.
    auto batch_size(size_t const new_batch_size) -> void
    {
        _batch_size = new_batch_size;
        if (_input.defined()) {
            _input.resize_({static_cast<int64_t>(_batch_size), _input.size(1)});
        }
    }

    auto operator()(SpinVector) -> float;

    auto time_poly() const -> std::pair<real_type, real_type>
    {
        return {_poly_time.mean(), std::sqrt(_poly_time.variance())};
    }

    auto time_psi() const -> std::pair<real_type, real_type>
    {
        return {_psi_time.mean(), std::sqrt(_psi_time.variance())};
    }

  private:
    /// Let `_poly` be `∑cᵢ|σᵢ⟩` where `i` runs from `0` to `N-1`.
    /// `forward_propagate_batch` calculates `∑cᵢ⟨σᵢ|ψ⟩` where `i` now runs from
    /// `n * batch_size` to `(n + 1) * batch_size`.
    auto forward_propagate_batch(size_t n) -> float;

    /// Let `_poly` be `∑cᵢ|σᵢ⟩` where `i` runs from `0` to `N-1`.
    /// `forward_propagate_rest` calculates `∑cᵢ⟨σᵢ|ψ⟩` where `i` now runs from
    /// `n * batch_size` to `n * batch_size + rest` (which should be equal to
    /// `N-1`).
    auto forward_propagate_rest(size_t n, size_t rest) -> float;
};

auto PolynomialState::operator()(SpinVector const input) -> float
{
    using MicroSecondsT =
        std::chrono::duration<real_type, std::chrono::microseconds::period>;
    // TODO(twesterhout): Add a lookup in the table shared between multiple
    // parallel Markov chains.
    // Upon construction of the state, we don't know the number of spins in the
    // system. We extract it from the input.
    if (!_input.defined() || _input.size(1) != input.size()) {
        _input = detail::make_f32_tensor(batch_size(), input.size());
    }
    // TODO(twesterhout): It'd be nice to time how much time this function
    // spends applying `_poly` to `input` and how much on forward propagation
    // through `_psi`.
    auto time_start = std::chrono::steady_clock::now();
    _poly(real_type{1}, input);
    auto time_interval =
        MicroSecondsT(std::chrono::steady_clock::now() - time_start);
    _poly_time(time_interval.count());

    time_start                = std::chrono::steady_clock::now();
    auto const size           = _poly.size();
    auto const number_batches = size / batch_size();
    auto const rest           = size % batch_size();
    auto       sum            = 0.0f;
    for (auto i = size_t{0}; i < number_batches; ++i) {
        sum += forward_propagate_batch(i);
    }
    if (rest != 0) { sum += forward_propagate_rest(number_batches, rest); }
    time_interval =
        MicroSecondsT(std::chrono::steady_clock::now() - time_start);
    _psi_time(time_interval.count());
    return sum;
}

auto PolynomialState::forward_propagate_batch(size_t const n) -> float
{
    TCM_ASSERT(_input.defined(),
               "You should allocate the storage before calling this function");
    TCM_ASSERT((n + 1) * _batch_size <= _poly.size(), "Index out of bounds");
    // Stores the `n`th batch of `_poly.vectors()` into `_input`.
    auto* data = _poly.vectors().data() + n * batch_size();
    detail::unpack_to_tensor(/*first=*/data, /*last=*/data + batch_size(),
                             /*destination=*/_input);
    // Forward propagates the batch through our network `_psi`.
    auto output = _psi(_input).view({-1});
    // Extracts the `n`th batch of `_poly.coefficients()`.
    auto coefficients = _poly.coefficients().slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(n * batch_size()),
        /*end=*/static_cast<int64_t>((n + 1) * batch_size()), /*step=*/1);
    // Computes the final result.
    // NOTE: This can't be optimised much because `_psi` is a user supplied
    // function which might return tensors of wrong shape.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

auto PolynomialState::forward_propagate_rest(size_t const n, size_t const rest)
    -> float
{
    TCM_ASSERT(_input.defined(),
               "You should allocate the storage before calling this function");
    TCM_ASSERT(rest < batch_size(), "Go use forward_propagate_batch instead");
    TCM_ASSERT(n * batch_size() + rest == _poly.size(),
               "Precondition violated");
    // Stores part of batch which we're given into `_input`.
    auto* data = _poly.vectors().data() + n * batch_size();
    detail::unpack_to_tensor(
        /*first=*/data, /*last=*/data + rest,
        /*destination=*/
        _input.slice(/*dim=*/0, /*start=*/0, /*end=*/static_cast<int64_t>(rest),
                     /*step=*/1));
    // Fills the remaining part of the batch with spin ups.
    _input.slice(/*dim=*/0, /*start=*/static_cast<int64_t>(rest),
                 /*end=*/static_cast<int64_t>(batch_size()),
                 /*step=*/1) = 1.0f;
    // Forward progates the batch through out network `_psi`. Only the first
    // `rest` components contain meaningful info.
    auto output = _psi(_input)
                      .slice(/*dim=*/0, /*start=*/0,
                             /*end=*/static_cast<int64_t>(rest), /*rest=*/1)
                      .view({-1});
    // Extracts part of the `n`th batch of `_poly.coefficients()`.
    auto coefficients = _poly.coefficients().slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(n * batch_size()),
        /*end=*/static_cast<int64_t>(n * batch_size() + rest), /*step=*/1);
    // Computes the final result.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

#if 0
class TargetStateImpl : public torch::nn::Module {

  private:
    std::shared_ptr<Machine>    _psi;
    ToDo                        _todo;
    std::shared_ptr<Polynomial> _poly;
    size_t                      _batch_size;

    template <class Type, size_t Dim> struct ForwardFn;
    friend struct ForwardFn<float, 1>;
    friend struct ForwardFn<float, 2>;

  public:
    TargetStateImpl(std::shared_ptr<Machine>    psi,
                    std::shared_ptr<Polynomial> poly, size_t const batch_size)
        : _psi{std::move(psi)}
        , _todo{}
        , _poly{std::move(poly)}
        , _batch_size{batch_size}
    {
        constexpr auto const* function =
            "TargetState(shared_ptr<Machine>, shared_ptr<Polynomial>)";
        if (TCM_UNLIKELY(_psi == nullptr)) {
            throw std::invalid_argument{std::string{function}
                                        + ": machine must not be nullptr"};
        }
        if (TCM_UNLIKELY(_poly == nullptr)) {
            throw std::invalid_argument{std::string{function}
                                        + ": polynomial must not be nullptr"};
        }
        _poly->clear();
    }

    TargetStateImpl(TargetStateImpl const&) = delete;
    TargetStateImpl(TargetStateImpl&&)      = delete;
    TargetStateImpl& operator=(TargetStateImpl const&) = delete;
    TargetStateImpl& operator=(TargetStateImpl&&) = delete;

    auto batch_size() const noexcept -> size_t { return _batch_size; }

    auto batch_size(size_t const new_batch_size) noexcept -> size_t
    {
        return _batch_size = new_batch_size;
    }

    auto operator()(torch::Tensor const& input) -> torch::Tensor
    {
        return forward(input);
    }

    auto forward(torch::Tensor const& input) -> torch::Tensor;
};

template <>
struct TargetStateImpl::ForwardFn<float, 1> {
    auto operator()(TargetStateImpl& self, torch::Tensor const& input) const
        -> torch::Tensor
    {
        TCM_ASSERT(
            input.type().scalarType() == torch::kFloat32,
            "Bug! This function should only be called with Float32 Tensors");
        TCM_ASSERT(input.dim() == 1,
                   "Bug! This function should only be called with 1D Tensors");
        return (*this)(self, input, SpinVector{input});
    }

    auto operator()(TargetStateImpl& self, torch::Tensor const& input,
                    SpinVector const spin) const -> torch::Tensor
    {
        TCM_ASSERT(
            input.type().scalarType() == torch::kFloat32,
            "Bug! This function should only be called with Float32 Tensors");
        TCM_ASSERT(input.dim() == 1,
                   "Bug! This function should only be called with 1D Tensors");

        self._todo.clear();
        self._todo.insert((*self._poly)(real_type{1}, spin));
        static_cast<void>(input);
        return self._todo.evaluate(*self._psi, self.batch_size());
    }
};

template <>
struct TargetStateImpl::ForwardFn<float, 2> {
    auto operator()(TargetStateImpl& self, torch::Tensor const& input) const
        -> torch::Tensor
    {
        TCM_ASSERT(
            input.type().scalarType() == torch::kFloat32,
            "Bug! This function should only be called with Float32 Tensors");
        TCM_ASSERT(input.dim() == 2,
                   "Bug! This function should only be called with 1D Tensors");
        self._todo.clear();
        auto accessor = input.template accessor<float, 2>();
        for (auto i = int64_t{0}; i < input.size(0); ++i) {
            self._todo.insert(
                (*self._poly)(real_type{1}, SpinVector{accessor[i]}),
                self._psi->search_fn());
            if (self._todo.size() >= self.batch_size()) {
                self._todo.evaluate_batch(*self._psi, self.batch_size());
            }
        }
        return self._todo.evaluate(*self._psi, self.batch_size());
    }
};

auto TargetStateImpl::forward(torch::Tensor const& input) -> torch::Tensor
{
    constexpr auto const* function = "TargetState::forward(Tensor const&)";
    torch::NoGradGuard    no_grad;

    auto const scalar_type = input.type().scalarType();
    if (TCM_UNLIKELY(scalar_type != torch::kFloat32))
        detail::error_wrong_type(function, scalar_type, torch::kFloat32);

    auto const dim = input.dim();
    switch (dim) {
    case 1: return ForwardFn<float, 1>{}(*this, input);
    case 2: return ForwardFn<float, 2>{}(*this, input);
    default: detail::error_wrong_dim(function, dim, 1, 2);
    } // end switch
}

TORCH_MODULE(TargetState);
#endif

using RandomGenerator = std::mt19937;

auto global_random_generator() -> RandomGenerator&;

class RandomFlipper {

  public:
    using index_type = unsigned;

    static constexpr index_type number_flips = 2;

    using value_type = std::array<index_type, number_flips>;

  private:
    std::vector<index_type>         _storage;
    gsl::span<index_type>           _ups;
    gsl::span<index_type>           _downs;
    gsl::not_null<RandomGenerator*> _generator;
    index_type                      _i;

  public:
    RandomFlipper(SpinVector       initial_spin,
                  RandomGenerator& generator = global_random_generator());

    RandomFlipper(RandomFlipper const&) = delete;
    RandomFlipper(RandomFlipper&&)      = default;
    RandomFlipper& operator=(RandomFlipper const&) = delete;
    RandomFlipper& operator=(RandomFlipper&&) = default;

    inline auto read() const TCM_NOEXCEPT -> value_type;
    inline auto next(bool accepted) -> void;

  private:
    auto shuffle() -> void;
    auto swap_accepted() TCM_NOEXCEPT -> void;
};

constexpr RandomFlipper::index_type RandomFlipper::number_flips;

auto RandomFlipper::read() const TCM_NOEXCEPT -> value_type
{
    using std::begin;
    using std::end;
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");

    value_type proposed;
    std::copy(begin(_ups) + _i, begin(_ups) + _i + n, begin(proposed));
    std::copy(begin(_downs) + _i, begin(_downs) + _i + n, begin(proposed) + n);
    return proposed;
}

auto RandomFlipper::next(bool const accepted) -> void
{
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");
    if (accepted) { swap_accepted(); }
    _i += n;
    if (_i + n > _ups.size() || _i + n > _downs.size()) {
        shuffle();
        _i = 0;
    }
}

RandomFlipper::RandomFlipper(SpinVector const initial_spin,
                             RandomGenerator& generator)
    : _storage(initial_spin.size())
    , _generator{std::addressof(generator)}
    , _i{0}
{
    using std::begin;
    using std::end;
    TCM_CHECK(initial_spin.size() >= number_flips, std::invalid_argument,
              fmt::format("requested number of spin-flips exceeds the number "
                          "of spins in the system: {} > {}.",
                          number_flips, initial_spin.size()));
    std::iota(begin(_storage), end(_storage), 0);
    auto const number_ups =
        static_cast<size_t>(std::partition(begin(_storage), end(_storage),
                                           [spin = initial_spin](auto const i) {
                                               return spin[i] == Spin::up;
                                           })
                            - begin(_storage));
    _ups   = gsl::span<index_type>{_storage.data(), number_ups};
    _downs = gsl::span<index_type>{_storage.data() + number_ups,
                                   _storage.size() - number_ups};
    TCM_CHECK(
        (_ups.size() < number_flips / 2 || _downs.size() < number_flips / 2),
        std::invalid_argument,
        fmt::format("initial spin is invalid. Given {} spins up and {} "
                    "spins down, it's impossible to perform {} "
                    "spin-flips and still preserve the magnetisation.",
                    _ups.size(), _downs.size(), number_flips));
    shuffle();
}

auto RandomFlipper::shuffle() -> void
{
    using std::begin;
    using std::end;
    std::shuffle(begin(_ups), end(_ups), *_generator);
    std::shuffle(begin(_downs), end(_downs), *_generator);
}

auto RandomFlipper::swap_accepted() TCM_NOEXCEPT -> void
{
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");
    for (auto i = _i; i < _i + n; ++i) {
        std::swap(_ups[i], _downs[i]);
    }
}

#if 0
class MarkovChain {

    using ForwardFn     = std::function<auto(SpinVector)->real_type>;
    using ProbabilityFn = std::function<auto(real_type, real_type)->real_type>;

  private:
    SpinVector                _spin;
    RandomFlipper             _flipper;
    ForwardFn                 _forward;
    ProbabilityFn             _prob;
    real_type                 _value;
    gsl::not_null<Generator*> _generator;

  public:
    MarkovChain(ForwardFn forward, ProbabilityFn prob, SpinVector const spin,
                RandomGenerator& generator)
        : _spin{spin}
        , _flipper{spin, generator}
        , _forward{std::move(forward)}
        , _prob{std::move(prob)}
        , _generator{std::addressof(generator)}
    {
        _value = _forward(_spin);
    }

    MarkovChain(MarkovChain const&) = default;
    MarkovChain(MarkovChain&&)      = default;
    MarkovChain& operator=(MarkovChain const&) = default;
    MarkovChain& operator=(MarkovChain&&) = default;

    constexpr auto const& read() const noexcept { return *_state; }

    auto next() -> void
    {
        auto const u =
            std::generate_canonical<R, std::numeric_limits<R>::digits>(
                *_generator);
        auto const flips = std::visit(
            [](auto const& x) noexcept { return x.read(); }, _flipper);
        auto const [log_quot_wf, cache] = _state->log_quot_wf(flips);
        auto const probability =
            std::min(R{1}, std::norm(std::exp(log_quot_wf)));
        if (u <= probability) {
            _state->update(flips, cache);
            std::visit([](auto& x) { x.next(true); }, _flipper);
        }
        else {
            std::visit([](auto& x) { x.next(false); }, _flipper);
        }
    }
};
#endif



TCM_NAMESPACE_END


#if defined(TCM_GCC)
#pragma GCC diagnostic pop
#endif
