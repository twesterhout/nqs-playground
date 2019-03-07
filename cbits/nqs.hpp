#pragma once

#include <boost/align/aligned_allocator.hpp>
#include <boost/config.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <flat_hash_map/bytell_hash_map.hpp>
#include <ska_sort.hpp>

#include <gsl/gsl-lite.hpp>

// #include <tbb/parallel_reduce.h>
// #include <tbb/blocked_range.h>

#include <expected.hpp>

#if defined(BOOST_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#    pragma GCC diagnostic ignored "-Wsign-promo"
#    pragma GCC diagnostic ignored "-Wswitch-default"
#    pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#    pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
#include <fmt/format.h>
#if defined(BOOST_GCC)
#    pragma GCC diagnostic pop
#endif

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
#include <omp.h>

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

/// Formatting of torch::ScalarType using fmtlib facilities
///
/// Used only for error reporting.
template <> struct fmt::formatter<torch::ScalarType> : formatter<string_view> {
    // parse is inherited from formatter<string_view>.

    template <typename FormatContext>
    auto format(torch::ScalarType const type, FormatContext& ctx)
    {
        // TODO(twesterhout): Us using c10 is probably not what PyTorch folks had
        // in mind... Suggestions are welcome
        return formatter<string_view>::format(::c10::toString(type), ctx);
    }
};

TCM_NAMESPACE_BEGIN

using std::size_t;
using std::uint16_t;
using std::uint64_t;

using real_type    = double;
using complex_type = std::complex<real_type>;

using torch::optional;
using torch::nullopt;

struct SplitTag {};
struct SerialTag {};
struct ParallelTag {};

enum class Spin : unsigned char {
    down = 0x00,
    up   = 0x01,
};

namespace detail {
struct UnsafeTag {};
constexpr UnsafeTag unsafe_tag;

/// Horizontally adds elements of a float4 vector.
///
/// Solution taken from https://stackoverflow.com/a/35270026
TCM_FORCEINLINE static auto hadd(__m128 const v) noexcept -> float
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/// Horizontally adds elements of a float8 vector.
TCM_FORCEINLINE static auto hadd(__m256 const v) noexcept -> float
{
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow         = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hadd(vlow);
}
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

inline auto make_i64_tensor(size_t n) -> torch::Tensor
{
    TCM_ASSERT(n <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
               "Integer overflow");
    auto out = torch::empty(
        {static_cast<int64_t>(n)},
        torch::TensorOptions().dtype(torch::kInt64).requires_grad(false));
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


// Errors {{{
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
    return fmt::format("wrong dimension {:d}; expected {:d}", dimension,
                       expected);
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

#define TCM_CHECK_TYPE(type, expected)                                         \
    TCM_CHECK(type == expected, std::domain_error,                             \
              ::fmt::format("wrong type {}; expected {}", type, expected))
// }}}


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
    constexpr SpinVector() noexcept : _data{} { _data.as_ints[0] = 0; _data.as_ints[1] = 0; }

    constexpr SpinVector(SpinVector const&) noexcept = default;
    constexpr SpinVector(SpinVector&&) noexcept      = default;
    constexpr SpinVector& operator=(SpinVector const&) noexcept = default;
    constexpr SpinVector& operator=(SpinVector&&) noexcept = default;

    explicit SpinVector(gsl::span<float const>);
    explicit SpinVector(gsl::span<float const>, detail::UnsafeTag) TCM_NOEXCEPT;

    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    explicit SpinVector(pybind11::array_t<float, ExtraFlags> const& spins);
    explicit SpinVector(pybind11::str str);

    explicit SpinVector(torch::Tensor const& spins);
    explicit SpinVector(torch::TensorAccessor<float, 1> accessor);

    inline constexpr auto        size() const noexcept -> unsigned;
    inline static constexpr auto max_size() noexcept -> unsigned;

    inline /*constexpr*/ auto magnetisation() const noexcept -> int;

    inline constexpr auto operator[](unsigned const i) const
        & TCM_NOEXCEPT -> Spin;
    inline constexpr auto operator[](unsigned const i) && TCM_NOEXCEPT -> Spin;
    inline constexpr auto operator[](unsigned const i)
        & TCM_NOEXCEPT -> SpinReference;

    inline constexpr auto at(unsigned const i) const& -> Spin;
    inline constexpr auto at(unsigned const i) && -> Spin;
    inline constexpr auto at(unsigned const i) & -> SpinReference;

    inline constexpr auto flip(unsigned i) TCM_NOEXCEPT -> void;

    template <size_t N>
    inline constexpr auto flipped(std::array<unsigned, N> is) const TCM_NOEXCEPT
        -> SpinVector;

    inline constexpr auto
    flipped(std::initializer_list<unsigned> is) const TCM_NOEXCEPT
        -> SpinVector;

    inline auto operator==(SpinVector const& other) const TCM_NOEXCEPT -> bool;
    inline auto operator!=(SpinVector const& other) const TCM_NOEXCEPT -> bool;

    inline auto operator<(SpinVector const& other) const TCM_NOEXCEPT -> bool;

    inline auto     hash() const noexcept -> size_t;
    inline explicit operator uint64_t() const;
    inline explicit operator std::string() const;

    inline auto copy_to(gsl::span<float> buffer) const TCM_NOEXCEPT -> void;

    auto numpy() const -> pybind11::array_t<float, pybind11::array::c_style>;
    auto tensor() const -> torch::Tensor;

    template <class Generator>
    static auto random(unsigned size, int magnetisation, Generator& generator)
        -> SpinVector;

    template <class Generator>
    static auto random(unsigned size, Generator& generator) -> SpinVector;

    auto key(detail::UnsafeTag) const TCM_NOEXCEPT -> uint64_t
    {
        TCM_ASSERT(size() <= 64, "Chain too long");
        return *reinterpret_cast<uint64_t const*>(_data.spin);
    }

    auto dump() const -> std::string
    {
        std::ostringstream msg;
        msg << size() << ": ";
        for (auto i = 0u; i < size(); ++i) {
            msg << static_cast<unsigned>(unsafe_at(i));
        }
        msg << " ";
        for (auto i = size(); i < max_size(); ++i) {
            msg << static_cast<unsigned>(unsafe_at(i));
        }
        return msg.str();
    }

    constexpr auto is_valid() const TCM_NOEXCEPT -> bool
    {
        for (auto i = size(); i < max_size(); ++i) {
            if (unsafe_at(i) != Spin::down) { return false; }
        }
        return true;
    }

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

    TCM_FORCEINLINE static auto
    load_u16_short(gsl::span<float const> buffer) TCM_NOEXCEPT -> uint16_t
    {
        TCM_ASSERT(buffer.size() < 16, "Range too long");
        TCM_ASSERT(is_valid_spin(buffer), "Invalid spin value");
        auto result = uint16_t{0};
        for (auto i = size_t{0}; i < buffer.size(); ++i) {
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

constexpr auto SpinVector::operator[](unsigned const i)
    && TCM_NOEXCEPT -> Spin
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

inline auto
SpinVector::copy_to(gsl::span<float> const buffer) const TCM_NOEXCEPT -> void
{
    TCM_ASSERT(buffer.size() == size(), "Wrong buffer size");
    auto spin2float = [](Spin const s) noexcept->float
    {
        return s == Spin::up ? 1.0f : -1.0f;
    };

    for (auto i = 0u; i < size(); ++i) {
        buffer[i] = spin2float((*this)[i]);
    }
}

inline SpinVector::SpinVector(gsl::span<float const> buffer,
                              detail::UnsafeTag /*unused*/) TCM_NOEXCEPT
{
    check_range(buffer, detail::unsafe_tag);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

template <int ExtraFlags, class = std::enable_if_t<
                              ExtraFlags & pybind11::array::c_style
                              || ExtraFlags & pybind11::array::f_style> /**/>
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
        using Param = Dist::param_type;
        spin._data.spin[chunks] =
            dist(generator, Param{0, static_cast<uint16_t>((1 << rest) - 1)})
            << (16 - rest);
    }

    spin._data.size = size;
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

TCM_NAMESPACE_END

namespace std {
template <> struct hash<::TCM_NAMESPACE::SpinVector> {
    auto operator()(::TCM_NAMESPACE::SpinVector const& spin) const noexcept
        -> size_t
    {
        return spin.hash();
    }
};
} // namespace std

TCM_NAMESPACE_BEGIN

namespace detail {
struct IdentityProjection {
    template <class T> constexpr decltype(auto) operator()(T&& x) const noexcept
    {
        return std::forward<T>(x);
    }
};

template <class RandomAccessIterator, class Projection = IdentityProjection>
auto unpack_to_tensor(RandomAccessIterator begin, RandomAccessIterator end,
                      Projection proj = IdentityProjection{}) -> torch::Tensor
{
    if (begin == end) return detail::make_f32_tensor(0);
    TCM_ASSERT(end - begin > 0, "Invalid range");
    auto const size         = static_cast<size_t>(end - begin);
    auto const number_spins = proj(*begin).size();
    TCM_ASSERT(std::all_of(begin, end,
                           [number_spins, &proj](auto const& x) {
                               return proj(x).size() == number_spins;
                           }),
               "Input range contains variable sized spin chains");
    auto out = detail::make_f32_tensor(static_cast<size_t>(size), number_spins);
    auto*      data = out.template data<float>();
    auto const ldim = out.stride(0);
    for (auto i = size_t{0}; i < size; ++i, ++begin, data += ldim) {
        proj(*begin).copy_to({data, number_spins});
    }
    return out;
}

template <class RandomAccessIterator>
auto unpack_to_tensor(RandomAccessIterator begin, RandomAccessIterator end,
                      torch::Tensor output)
{
#if 1
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
        begin->copy_to({data, number_spins});
    }
#endif
}
} // namespace detail

// [QuantumState] {{{
class QuantumState
    : public ska::bytell_hash_map<SpinVector, complex_type> {
  private:
    using base =
        ska::bytell_hash_map<SpinVector, complex_type>;
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
        auto& c = static_cast<base&>(*this)[value.second];
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
    QuantumState _current;
    QuantumState _old;
    /// Hamiltonian which knows how to perform `|ψ⟩ += c * H|σ⟩`.
    std::shared_ptr<Heisenberg const> _hamiltonian;
    /// List of terms.
    std::vector<Term> _terms;
    /// The result of applying the polynomial to a state `|ψ⟩`
    /// can be written as ∑cᵢ|σᵢ⟩. `_basis` is the set {|σᵢ⟩}.
    std::vector<SpinVector> _basis;
    /// The result of applying the polynomial to a state `|ψ⟩`
    /// can be written as ∑cᵢ|σᵢ⟩. `_coeffs` is the set {cᵢ}.
    torch::Tensor _coeffs;

  public:
    /// Constructs the polynomial given the hamiltonian and a list or terms.
    Polynomial(std::shared_ptr<Heisenberg const> hamiltonian,
               std::vector<Term>                 terms);

    Polynomial(Polynomial const&) = default;
    Polynomial(Polynomial &&) = default;
    Polynomial& operator=(Polynomial const&) = delete;
    Polynomial& operator=(Polynomial &&) = delete;

    Polynomial(Polynomial const& other, SplitTag)
        : _current{}
        , _old{}
        , _hamiltonian{other._hamiltonian}
        , _terms{other._terms}
        , _basis{}
        , _coeffs{}
    {
        auto size = std::max(other._current.size(), other._old.size());
        _current.reserve(size);
        _old.reserve(size);
        _basis.reserve(other._basis.capacity());
        if (other._coeffs.defined()) {
            _coeffs = detail::make_f32_tensor(
                static_cast<size_t>(other._coeffs.size(0)));
        }
    }

    inline auto degree() const noexcept -> size_t;
    inline auto size() const noexcept -> size_t;
    inline auto clear() noexcept -> void;

    /// Applies the polynomial to state `|ψ⟩ = coeff * |spin⟩`.
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
// [Polynomial] }}}

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

// [VarAccumulator] {{{
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

    constexpr auto mean() const -> value_type
    {
        TCM_CHECK(_count > 0, std::runtime_error,
                  fmt::format("mean of 0 samples is not defined"));
        return _mean;
    }

    constexpr auto variance() const -> value_type
    {
        TCM_CHECK(_count > 1, std::runtime_error,
                  fmt::format("sample variance of {} samples is not defined",
                              _count));
        return _M2 / (_count - 1);
    }
};

static_assert(std::is_trivially_copyable<VarAccumulator>::value, "");
static_assert(std::is_trivially_destructible<VarAccumulator>::value, "");
// }}}

// [PolynomialState] {{{
namespace detail {
class _PolynomialState {
  public:
    using ForwardT = std::function<auto(torch::Tensor const&)->torch::Tensor>;

  private:
    Polynomial     _poly;
    size_t         _batch_size;
    VarAccumulator _poly_time;
    VarAccumulator _psi_time;

    struct Body;

  public:
    _PolynomialState(Polynomial poly, size_t const batch_size) noexcept
        : _poly{std::move(poly)}
        , _batch_size{batch_size}
        , _poly_time{}
        , _psi_time{}
    {
        static_assert(std::is_nothrow_move_constructible<Polynomial>::value,
                      "");
    }

    _PolynomialState(_PolynomialState const&) = delete;
    _PolynomialState(_PolynomialState&&)      = delete;
    _PolynomialState& operator=(_PolynomialState const&) = delete;
    _PolynomialState& operator=(_PolynomialState&&) = delete;

    /// Returns the batch size.
    constexpr auto batch_size() const noexcept -> size_t;

    /// Updates the batch size used internally for forward propagation. Note
    /// that this function may allocate and is thus not marked `noexcept`.
    constexpr auto batch_size(size_t) noexcept -> void;

    /// Runs forward propagation.
    auto operator()(SpinVector, ForwardT const&, std::string const&) -> float;

    auto time_poly() const -> std::pair<real_type, real_type>;
    auto time_psi() const -> std::pair<real_type, real_type>;
};

constexpr auto _PolynomialState::batch_size() const noexcept -> size_t
{
    return _batch_size;
}

constexpr auto
_PolynomialState::batch_size(size_t const new_batch_size) noexcept -> void
{
    _batch_size = new_batch_size;
}
} // namespace detail


class PolynomialState : private detail::_PolynomialState {
  private:
    using base = detail::_PolynomialState;
    using FunctionT = base::ForwardT;

    FunctionT   _psi;
    std::string _filename;

  public:
    PolynomialState(FunctionT psi, Polynomial poly, size_t const batch_size,
                    std::string const& filename)
        : base{std::move(poly), batch_size}
        , _psi{std::move(psi)}
        , _filename{filename}
    {}

    using base::batch_size;
    using base::time_poly;
    using base::time_psi;

    auto operator()(SpinVector x) -> float
    {
        return static_cast<base&>(*this)(x, _psi, _filename);
    }
};

#if 0
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

    // TODO
    // auto operator()(torch::Tensor const&) -> torch::Tensor;

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
#endif
// }}}

// [Random] {{{
using RandomGenerator = std::mt19937;

auto global_random_generator() -> RandomGenerator&;
// }}}

// [RandomFlipper] {{{
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
    inline auto swap_accepted() TCM_NOEXCEPT -> void;
};
// }}}

// [RandomFlipper.implementation] {{{
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

inline auto RandomFlipper::swap_accepted() TCM_NOEXCEPT -> void
{
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");
    for (auto i = _i; i < _i + n; ++i) {
        std::swap(_ups[i], _downs[i]);
    }
}
// }}}

struct DefaultProbFn {
    using RealT = real_type;

    auto operator()(RealT current, RealT suggested) const noexcept -> RealT
    {
        current   = current * current;
        suggested = suggested * suggested;
        if (current <= suggested) return real_type{1};
        return suggested / current;
    }
};

// This is very similar to std::unique from libc++ except for the else
// branch which combines equal values.
template <class ForwardIterator, class EqualFn, class MergeFn>
auto compress(ForwardIterator first, ForwardIterator last, EqualFn equal,
              MergeFn merge) -> ForwardIterator
{
    first =
        std::adjacent_find<ForwardIterator,
                           typename std::add_lvalue_reference<EqualFn>::type>(
            first, last, equal);
    if (first != last) {
        auto i = first;
        for (++i; ++i != last;) {
            if (!equal(*first, *i)) { *++first = std::move(*i); }
            else {
                merge(*first, *i);
            }
        }
        ++first;
    }
    return first;
}

// [ChainState] {{{
struct alignas(32) ChainState {
    SpinVector spin;  ///< Spin configuration σ
    real_type  value; ///< Wave function ψ(σ)
    size_t     count; ///< Number of times this state has been visited

    // Simple constructor to make `emplace` happy.
    constexpr ChainState(SpinVector s, real_type v, size_t n) noexcept
        : spin{s}, value{v}, count{n}
    {}

    ChainState()                                     = delete;
    constexpr ChainState(ChainState const&) noexcept = default;
    constexpr ChainState(ChainState&&) noexcept      = default;
    constexpr ChainState& operator=(ChainState const&) noexcept = default;
    constexpr ChainState& operator=(ChainState&) noexcept = default;

    /// Merges `other` into this. This amounts to just adding together the
    /// `count`s. In DEBUG mode, however, we also make sure that only states
    /// with the same `spin` and `value` attributes can be merged.
    TCM_CONSTEXPR auto merge(ChainState const& other) TCM_NOEXCEPT -> void
    {
        auto const isclose = [](auto const a, auto const b) noexcept->bool
        {
            using std::abs;
            using std::max;
            constexpr auto atol = 1.0E-5;
            constexpr auto rtol = 1.0E-3;
            return abs(a - b) <= atol + rtol * max(abs(a), abs(b));
        };
        TCM_ASSERT(spin == other.spin,
                   "only states with the same spin can be merged");
        TCM_ASSERT(isclose(value, other.value),
                   "Different forward passes with the same input should "
                   "produce the same results");
        count += other.count;
    }
};

static_assert(sizeof(ChainState) == 32, "");
static_assert(std::is_trivially_copyable<ChainState>::value, "");
static_assert(std::is_trivially_destructible<ChainState>::value, "");

#define TCM_MAKE_OPERATOR_USING_KEY(op)                                        \
    inline auto operator op(ChainState const& x, ChainState const& y)          \
        TCM_NOEXCEPT->bool                                                     \
    {                                                                          \
        TCM_ASSERT(x.spin.size() == y.spin.size(),                             \
                   "States corresponding to different system sizes can't be "  \
                   "compared");                                                \
        TCM_ASSERT(x.spin.size() <= 64,                                        \
                   "Longer spin chains are not (yet) supported");              \
        return x.spin.key(detail::unsafe_tag)                                  \
            op y.spin.key(detail::unsafe_tag);                                 \
    }

TCM_MAKE_OPERATOR_USING_KEY(==)
TCM_MAKE_OPERATOR_USING_KEY(!=)
TCM_MAKE_OPERATOR_USING_KEY(<)
TCM_MAKE_OPERATOR_USING_KEY(>)
TCM_MAKE_OPERATOR_USING_KEY(<=)
TCM_MAKE_OPERATOR_USING_KEY(>=)

#undef TCM_MAKE_OPERATOR_USING_KEY
// }}}

// [ChainResult] {{{
struct ChainResult {
    using SamplesT =
        std::vector<ChainState, boost::alignment::aligned_allocator<
                                    ChainState, alignof(ChainState)> /**/>;

  private:
    SamplesT _samples;

  public:
    ChainResult() noexcept              = default;
    ChainResult(ChainResult const&)     = delete;
    ChainResult(ChainResult&&) noexcept = default;
    ChainResult& operator=(ChainResult const&) = delete;
    ChainResult& operator=(ChainResult&&) noexcept = default;

    /// \precondition `samples` must be sorted!
    ChainResult(SamplesT samples) noexcept : _samples{std::move(samples)} {}

    constexpr auto samples() const & noexcept -> SamplesT const&
    {
        return _samples;
    }

    auto samples() && noexcept -> SamplesT { return std::move(_samples); }

    auto to_tensors() const -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    {
        return {extract_vectors(), extract_values(), extract_count()};
    }

  private:
    auto extract_vectors() const -> torch::Tensor;
    auto extract_values() const -> torch::Tensor;
    auto extract_count() const -> torch::Tensor;
};

auto merge(ChainResult const& x, ChainResult const& y) -> ChainResult;
// }}}

// [MarkovChain] {{{
template <class ForwardFn, class ProbabilityFn> class MarkovChain {
    using SamplesT = ChainResult::SamplesT;

  private:
    SamplesT                        _samples;
    RandomFlipper                   _flipper;
    ForwardFn                       _forward;
    ProbabilityFn                   _prob;
    gsl::not_null<RandomGenerator*> _generator;
    bool                            _is_sorted;
    bool                            _is_recording;
    size_t                          _accepted;
    size_t                          _count;

  public:
    MarkovChain(ForwardFn forward, ProbabilityFn prob, SpinVector const spin,
                RandomGenerator& generator)
        : _samples{}
        , _flipper{spin, generator}
        , _forward{std::move(forward)}
        , _prob{std::move(prob)}
        , _generator{std::addressof(generator)}
        , _is_sorted{true}
        , _is_recording{false}
        , _accepted{1}
        , _count{1}
    {
        TCM_CHECK(
            spin.size() <= 64, std::runtime_error,
            fmt::format("Sorry, but such long spin chains ({}) are not (yet) "
                        "supported. The greatest supported length is {}",
                        spin.size(), 64));
        _samples.emplace_back(spin, _forward(spin), 1u);
    }

    MarkovChain(MarkovChain const&) = default;
    MarkovChain(MarkovChain&&)      = default;
    MarkovChain& operator=(MarkovChain const&) = default;
    MarkovChain& operator=(MarkovChain&&) = default;

    auto read() const noexcept -> ChainState const&
    {
        TCM_ASSERT(!_samples.empty(), "Bug! Use after `release`");
        return _samples.back();
    }

    auto start_recording() noexcept -> void
    {
        if (_is_recording) return;
        TCM_ASSERT(_samples.size() == 1, "Bug!");
        _is_recording         = true;
        _samples.back().count = 1;
    }

    auto next() -> void;

    auto release(bool sorted = true) && -> SamplesT;

  private:
    TCM_FORCEINLINE auto current_spin() const TCM_NOEXCEPT -> SpinVector
    {
        return _samples.back().spin;
    }

    TCM_FORCEINLINE auto current_value() const TCM_NOEXCEPT -> real_type
    {
        return _samples.back().value;
    }

    auto sort() -> void;
    auto compress() -> void;
};

template <class ForwardFn, class ProbabilityFn>
TCM_NOINLINE auto MarkovChain<ForwardFn, ProbabilityFn>::next() -> void
{
    TCM_ASSERT(!_samples.empty(), "Bug! Use after `release`");
    auto const u = std::generate_canonical<
        real_type, std::numeric_limits<real_type>::digits>(*_generator);
    auto       spin        = current_spin().flipped(_flipper.read());
    auto const value       = _forward(spin);
    auto const probability = _prob(current_value(), value);
    if (u <= probability) {
        if (_is_recording) {
            _samples.emplace_back(spin, value, 1u);
            _is_sorted = false;
        }
        else {
            TCM_ASSERT(_samples.size() == 1, "Bug!");
            _samples.back() = ChainState{spin, value, 1u};
        }
        _flipper.next(true);
    }
    else {
        ++(_samples.back().count);
        _flipper.next(false);
    }
}

template <class ForwardFn, class ProbabilityFn>
TCM_NOINLINE auto MarkovChain<ForwardFn, ProbabilityFn>::sort() -> void
{
    TCM_ASSERT(!_samples.empty(), "Bug! Use after `release`");
    TCM_ASSERT(!_is_sorted, "Bug! Why are you trying to sort twice?");
    auto const number_spins = current_spin().size();
    TCM_CHECK(number_spins <= 64, std::runtime_error,
              fmt::format("Sorry, but such long spin chains ({}) are not (yet) "
                          "supported. The greatest supported length is {}",
                          number_spins, 64));
    ska_sort(std::begin(_samples), std::end(_samples), [](auto const& x) {
        return x.spin.key(
            detail::unsafe_tag /*Yes, we have checked that size() <= 64*/);
    });
}

template <class ForwardFn, class ProbabilityFn>
auto MarkovChain<ForwardFn, ProbabilityFn>::compress() -> void
{
    TCM_ASSERT(!_samples.empty(), "Bug! Use after `release`");
    auto const pred  = [](auto const& x, auto const& y) { return x == y; };
    auto const merge = [](auto& acc, auto const& x) { acc.merge(x); };

    // This is very similar to std::unique from libc++ except for the else
    // branch which accumulates the count.
    auto       first = std::begin(_samples);
    auto const last  = std::end(_samples);
    first            = std::adjacent_find(first, last, pred);
    if (first != last) {
        auto i = first;
        for (++i; ++i != last;) {
            if (!pred(*first, *i)) { *++first = std::move(*i); }
            else {
                merge(*first, *i);
            }
        }
        ++first;
    }
    _samples.erase(first, last);
}

template <class ForwardFn, class ProbabilityFn>
auto MarkovChain<ForwardFn, ProbabilityFn>::release(bool sorted) && -> SamplesT
{
    TCM_ASSERT(!_samples.empty(), "Bug! Use after `release`");
    if (!sorted || _is_sorted) return std::move(_samples);
    sort();
    compress();
    _is_sorted = true;
    return std::move(_samples);
}
// }}}

auto random_spin(size_t size, int magnetisation, RandomGenerator& generator)
    -> SpinVector;


struct Options {
    unsigned                           number_spins;
    int                                magnetisation;
    std::tuple<size_t, size_t, size_t> steps;
};

TCM_NAMESPACE_END

namespace fmt {
template <> struct formatter<::TCM_NAMESPACE::Options> {
    template <typename ParseContext> constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(::TCM_NAMESPACE::Options const& options, FormatContext& ctx)
    {
        return format_to(
            ctx.begin(),
            "Options(number_spins={}, magnetisation={}, steps=({}, {}, {}))",
            options.number_spins, options.magnetisation,
            std::get<0>(options.steps), std::get<1>(options.steps),
            std::get<2>(options.steps));
    }
};
} // namespace fmt

TCM_NAMESPACE_BEGIN

// [sample_some] {{{
namespace detail {
template <class Function, class = void> struct FunctionWrapperHelper;

template <class Function>
struct FunctionWrapperHelper<
    Function,
    std::enable_if_t<std::is_lvalue_reference<Function>::value> /**/> {
    using type         = std::remove_const_t<std::remove_reference_t<Function>>;
    using wrapper_type = std::reference_wrapper<type const>;

    TCM_FORCEINLINE constexpr auto operator()(type const& x) -> wrapper_type
    {
        return std::cref(x);
    }
};

template <class Function>
struct FunctionWrapperHelper<
    Function,
    std::enable_if_t<std::is_rvalue_reference<Function>::value> /**/> {
    using type         = std::remove_const_t<std::remove_reference_t<Function>>;
    using wrapper_type = type;

    TCM_FORCEINLINE constexpr auto operator()(type&& x) -> wrapper_type
    {
        return static_cast<type&&>(x);
    }
};
} // namespace detail

template <class ForwardFn>
auto sample_some(ForwardFn&& psi, Options const& options,
                 optional<SpinVector> spin) -> ChainResult
{
    using Helper         = detail::FunctionWrapperHelper<ForwardFn&&>;
    auto&      generator = global_random_generator();
    auto const initial_spin =
        spin.has_value() ? *spin
                         : random_spin(options.number_spins,
                                       options.magnetisation, generator);
    MarkovChain<typename Helper::wrapper_type, DefaultProbFn> chain{
        Helper{}(std::forward<ForwardFn>(psi)), DefaultProbFn{}, initial_spin,
        generator};
    auto i = size_t{0};
    for (; i < std::get<0>(options.steps); i += std::get<2>(options.steps)) {
        chain.next();
    }
    chain.start_recording();
    for (; i < std::get<1>(options.steps); i += std::get<2>(options.steps)) {
        chain.next();
    }
    return ChainResult{std::move(chain).release()};
}
// }}}

// [parallel_sample_some] {{{
namespace detail {
template <class Iterator, class Body>
auto reduce_impl(SerialTag, Iterator begin, Iterator end, Body& body) noexcept
    -> std::exception_ptr
{
    static_assert(std::is_nothrow_move_constructible<Iterator>::value,
                  "Iterator's move constructor should not throw");
    try {
        body(begin, end);
        return nullptr;
    }
    catch (...) {
        return std::current_exception();
    }
}

template <class Iterator, class Body>
auto reduce_impl(ParallelTag, Iterator begin, Iterator end, Body& this_body,
                 size_t const grain_size) noexcept -> std::exception_ptr
{
    static_assert(
        std::is_nothrow_constructible<Body, Body const&, SplitTag>::value,
        "Body's \"split\"-constructor should not throw");
    static_assert(noexcept(begin + (end - begin) / 2),
                  "Iterator's operator- and operator+ should not throw");
    static_assert(std::is_nothrow_copy_constructible<Iterator>::value,
                  "Iterator's copy constructor should not throw");

    auto size = end - begin;
    if (static_cast<size_t>(size) <= grain_size) {
        return reduce_impl(SerialTag{}, begin, end, this_body);
    }

    auto const         middle = begin + (size + 1) / 2;
    Body               other_body{this_body, SplitTag{}};
    std::exception_ptr this_err;
    std::exception_ptr other_err;

#pragma omp task default(none) firstprivate(begin, middle)                     \
    shared(other_body, other_err)
    other_err =
        reduce_impl(ParallelTag{}, begin, middle, other_body, grain_size);

// #pragma omp task shared(this_body, this_err)
    this_err = reduce_impl(ParallelTag{}, middle, end, this_body, grain_size);

#pragma omp taskwait
    if (TCM_UNLIKELY(this_err != nullptr)) { return this_err; }
    if (TCM_UNLIKELY(other_err != nullptr)) { return other_err; }
    try {
        this_body.join(other_body);
        return nullptr;
    }
    catch (...) {
        return std::current_exception();
    }
}
} // namespace detail

template <class Iterator, class Body>
auto parallel_reduce(Iterator begin, Iterator end, Body& body,
                     size_t grain_size = 1, int const num_threads = -1) -> void
{
    TCM_CHECK(grain_size >= 1, std::invalid_argument,
              fmt::format("invalid grain_size {}; expected >= 1", grain_size));
    std::exception_ptr err;
    if (static_cast<size_t>(end - begin) <= grain_size) {
        err = detail::reduce_impl(SerialTag{}, begin, end, body);
    }
    else {
#pragma omp parallel
#pragma omp single nowait
        err = detail::reduce_impl(ParallelTag{}, begin, end, body, grain_size);
    }
    if (TCM_UNLIKELY(err != nullptr)) { std::rethrow_exception(err); }
}

template <class ForwardFn>
auto parallel_sample_some(size_t const number_chains, ForwardFn const& psi,
                          Options const& options) -> ChainResult
{
    struct Body {
      private:
        ForwardFn const& _fn;
        Options const&   _options;
        ChainResult      _result;

      public:
        Body()                = delete;
        Body(Body const&)     = delete;
        Body(Body&&) noexcept = default;
        Body& operator=(Body const&) = delete;
        Body& operator=(Body&&) noexcept = default;

        Body(ForwardFn const& fn, Options const& options) noexcept
            : _fn{fn}, _options{options}, _result{}
        {}

        Body(Body const& other, SplitTag) noexcept
            : Body{other._fn, other._options}
        {}

        auto operator()(size_t const begin, size_t const end) -> void
        {
            TCM_CHECK(end - begin == 1, std::runtime_error,
                      fmt::format("range has invalid size {}; expected 1",
                                  end - begin));
            _result = sample_some(_fn, _options, nullopt);
        }

        auto join(Body const& other) -> void
        {
            _result = merge(_result, other._result);
        }

        explicit operator ChainResult() && noexcept
        {
            return std::move(_result);
        }
    };

    Body total{psi, options};
    parallel_reduce(size_t{0}, number_chains, total, 1);
    return static_cast<ChainResult>(std::move(total));
}
// }}}

TCM_NAMESPACE_END
