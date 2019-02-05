#pragma once

#if defined(__clang__)
#    define TCM_CLANG                                                          \
        (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#    define TCM_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#    define TCM_GCC                                                            \
        (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#    define TCM_ASSUME(cond)                                                   \
        do {                                                                   \
            if (!(cond)) __builtin_unreachable();                              \
        } while (false)
#elif defined(_MSV_VER)
#    define TCM_MSVC _MSV_VER
#    define TCM_ASSUME(cond)                                                   \
        do {                                                                   \
        } while (false)
#else
#    error "Unsupported compiler."
#endif

#if defined(WIN32) || defined(_WIN32)
#    define TCM_EXPORT __declspec(dllexport)
#    define TCM_NOINLINE __declspec(noinline)
#    define TCM_FORCEINLINE __forceinline inline
#else
#    define TCM_EXPORT __attribute__((visibility("default")))
#    define TCM_NOINLINE __attribute__((noinline))
#    define TCM_FORCEINLINE __attribute__((always_inline)) inline
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

#include <pybind11/numpy.h>
#include <flat_hash_map/bytell_hash_map.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <immintrin.h>

TCM_NAMESPACE_BEGIN

using std::size_t;
using std::uint16_t;
using std::uint64_t;

enum class Spin : unsigned char {
    down = 0x00,
    up   = 0x01,
};

namespace detail {
struct UnsafeTag {};
constexpr UnsafeTag unsafe_tag;
} // namespace detail

class SpinVector {
  public:
    constexpr SpinVector() noexcept : _data{} {}
    constexpr SpinVector(SpinVector const&) noexcept = default;
    constexpr SpinVector(SpinVector&&) noexcept      = default;
    constexpr SpinVector& operator=(SpinVector const&) noexcept = default;
    constexpr SpinVector& operator=(SpinVector&&) noexcept = default;

  private:
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

  public:
    template <class IterB, class IterE>
    SpinVector(IterB begin, IterE end) TCM_NOEXCEPT
    {
        _data.as_ints = _mm_set1_epi32(0);
        auto i        = 0u;
        for (; begin != end; ++begin, ++i) {
            auto const s = *begin;
            TCM_ASSERT(s == -1 || s == +1, "Invalid spin.");
            TCM_ASSERT(i < max_size(), "Chain too long.");
            unsafe_at(i) = (s == 1) ? Spin::up : Spin::down;
        }
        _data.size = static_cast<uint16_t>(i);
    }

    template <class T> SpinVector(pybind11::array_t<T> spins)
    {
        static_assert(
            std::is_signed<T>::value || std::is_floating_point<T>::value,
            "Only signed integral and floating point types are supported");
        auto       access = spins.template unchecked<1>();
        auto const n      = spins.shape(0);
        if (n > max_size()) {
            throw std::invalid_argument{"Spin chain too long: max size of "
                                        + std::to_string(max_size())
                                        + " exceeded."};
        }
        _data.as_ints = _mm_set1_epi32(0);
        _data.size    = static_cast<std::uint16_t>(n);
        for (auto i = 0u; i < static_cast<unsigned>(n); ++i) {
            auto const s = access(i);
            if (s != -1 && s != +1) {
                throw std::domain_error{
                    "Invalid spin: expected one of {-1, +1}, but got "
                    + std::to_string(s) + "."};
            }
            unsafe_at(i) = (s == 1) ? Spin::up : Spin::down;
        }
    }

    SpinVector(std::initializer_list<int> spins) TCM_NOEXCEPT
        : SpinVector{spins.begin(), spins.end()}
    {}

  private:
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
            TCM_ASSERT(0 <= n && n < 16, "Index out of bounds.");
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

    constexpr auto unsafe_at(unsigned const i) noexcept -> SpinReference
    {
        return SpinReference{_data.spin[i / 16u], i % 16u};
    }

    constexpr auto unsafe_at(unsigned const i) const noexcept -> Spin
    {
        return static_cast<Spin>(get_bit(_data.spin[i / 16u], i % 16u));
    }

  public:
    constexpr auto size() const noexcept -> unsigned { return _data.size; }

    static constexpr auto max_size() noexcept -> unsigned
    {
        return 8 * sizeof(_data.spin);
    }

    constexpr auto operator[](unsigned const i) const TCM_NOEXCEPT -> Spin
    {
        TCM_ASSERT(i < size(), "Index out of bounds.");
        return unsafe_at(i);
    }

    constexpr auto operator[](unsigned const i) TCM_NOEXCEPT -> SpinReference
    {
        TCM_ASSERT(i < size(), "Index out of bounds.");
        return unsafe_at(i);
    }

    constexpr auto at(unsigned const i) const -> Spin
    {
        if (i >= size()) {
            throw std::invalid_argument{
                "Index out of bounds: " + std::to_string(i)
                + ", but the spin configuration has only "
                + std::to_string(size()) + " spins."};
        }
        return unsafe_at(i);
    }

    constexpr auto at(unsigned const i) -> SpinReference
    {
        if (i >= size()) {
            throw std::invalid_argument{
                "Index out of bounds: " + std::to_string(i)
                + ", but the spin configuration has only "
                + std::to_string(size()) + " spins."};
        }
        return unsafe_at(i);
    }

    constexpr auto flip(unsigned const i) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(i < size(), "Index out of bounds.");
        auto const chunk = i / 16u;
        auto const rest  = i % 16u;
        flip_bit(_data.spin[chunk], rest);
    }

    constexpr auto
    flipped(std::initializer_list<unsigned> is) const TCM_NOEXCEPT -> SpinVector
    {
        SpinVector temp{*this};
        for (auto const i : is) {
            temp.flip(i);
        }
        return temp;
    }

    auto operator==(SpinVector const& other) const TCM_NOEXCEPT -> bool
    {
        TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
        TCM_ASSERT(other.is_valid(), "SpinVector is in an invalid state");
        return _mm_movemask_epi8(_data.as_ints == other._data.as_ints)
               == 0xFFFF;
    }

    auto operator!=(SpinVector const other) const TCM_NOEXCEPT -> bool
    {
        TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
        TCM_ASSERT(other.is_valid(), "SpinVector is in an invalid state");
        return _mm_movemask_epi8(_data.as_ints == other._data.as_ints)
               != 0xFFFF;
    }

    auto hash() const noexcept -> std::size_t
    {
        static_assert(sizeof(_data.as_ints[0]) == sizeof(size_t), "");

        auto const hash_uint64 = [](std::uint64_t x) noexcept->std::uint64_t
        {
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EB;
            x = x ^ (x >> 31);
            return x;
        };

        auto const hash_combine = [hash_uint64](
                                      std::uint64_t seed,
                                      std::uint64_t x) noexcept->std::uint64_t
        {
            seed ^= hash_uint64(x) + std::uint64_t{0x9E3779B9} + (seed << 6)
                    + (seed >> 2);
            return seed;
        };

        return hash_combine(
            hash_uint64(static_cast<uint64_t>(_data.as_ints[0])),
            hash_uint64(static_cast<uint64_t>(_data.as_ints[1])));
    }

    explicit operator std::uint64_t() const
    {
        TCM_ASSERT(is_valid(), "SpinVector is in an invalid state");
        if (size() > 64) {
            throw std::overflow_error{
                "Spin chain is too long (" + std::to_string(size())
                + " spins) to be converted to a 64-bit integer."};
        }
        auto const x = (static_cast<uint64_t>(_data.spin[0]) << 48u)
                       + (static_cast<uint64_t>(_data.spin[1]) << 32u)
                       + (static_cast<uint64_t>(_data.spin[2]) << 16u)
                       + static_cast<uint64_t>(_data.spin[3]);
        return x >> (64u - size());
    }

    explicit operator std::string() const
    {
        std::string s(size(), 'X');
        for (auto i = 0u; i < size(); ++i) {
            s[i] = ((*this)[i] == Spin::up) ? '1' : '0';
        }
        return s;
    }

    template <class T> auto copy_to(T* buffer) const TCM_NOEXCEPT -> void
    {
        auto spin2float = [](Spin const s) noexcept->T
        {
            return s == Spin::up ? T{1} : T{-1};
        };

        for (auto i = 0u; i < size(); ++i) {
            buffer[i] = spin2float((*this)[i]);
        }
    }

    template <class Float,
              class = std::enable_if_t<std::is_floating_point<Float>::value>>
    auto numpy(pybind11::array_t<Float, pybind11::array::c_style> out) const
    {
        auto access = out.template mutable_unchecked<1>();
        if (out.shape(0) != static_cast<pybind11::ssize_t>(size())) {
            throw std::invalid_argument{
                "Output array has wrong shape: expected ("
                + std::to_string(size()) + ",), but got ("
                + std::to_string(out.shape(0)) + ",)."};
            return out;
        }

        auto spin2float = [](Spin const s) noexcept->Float
        {
            return s == Spin::up ? Float{1} : Float{-1};
        };

        for (auto i = 0u; i < size(); ++i) {
            access(i) = spin2float((*this)[i]);
        }
        return out;
    }

  private:
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
};

namespace detail {
struct SpinHasher {
    auto operator()(SpinVector const& x) const noexcept { return x.hash(); }
};
} // namespace detail

class QuantumState {
  public:
    using map_type =
        ska::bytell_hash_map<SpinVector, long double, detail::SpinHasher>;
    using value_type = map_type::value_type;

  private:
    map_type _table;

  public:
    QuantumState(std::size_t const number_buckets = 0) : _table{number_buckets}
    {}

    QuantumState(QuantumState const&) = default;
    QuantumState(QuantumState&&)      = default;
    QuantumState& operator=(QuantumState const&) = delete;
    QuantumState& operator=(QuantumState&&) = default;

    auto operator+=(std::pair<long double, SpinVector> const& value)
        -> QuantumState&
    {
        _table[value.second] += value.first;
        return *this;
    }

    template <class Key> auto& operator[](Key&& key)
    {
        return _table[std::forward<Key>(key)];
    }

    auto begin() const { return _table.cbegin(); }
    auto end() const { return _table.cend(); }
    auto size() const noexcept -> size_t { return _table.size(); }
    auto reserve(size_t const size) -> void { _table.reserve(size); }
    auto clear() -> void { _table.clear(); }

    template <class Key, class... Ts> auto emplace(Key&& key, Ts&&... xs)
    {
        return _table.emplace(std::forward<Key>(key), std::forward<Ts>(xs)...);
    }
};

/// \brief Represents the Heisenberg Hamiltonian.
class Heisenberg {
  public:
    using edge_type = std::pair<unsigned, unsigned>;

  private:
    std::vector<edge_type> _edges;
    long double            _coupling;

  public:
    Heisenberg(std::vector<edge_type> edges,
               long double const      coupling = 1.0l) TCM_NOEXCEPT
        : _edges{std::move(edges)}
        , _coupling{coupling}
    {}

    /// Copy and Move constructors/assignments
    Heisenberg(Heisenberg const&)     = default;
    Heisenberg(Heisenberg&&) noexcept = default;
    Heisenberg& operator=(Heisenberg const&) = delete;
    Heisenberg& operator=(Heisenberg&&) noexcept = default;

    auto                  size() const noexcept { return _edges.size(); }
    constexpr auto        coupling() const noexcept { return _coupling; }
    constexpr auto const& edges() const noexcept { return _edges; }

    constexpr auto coupling(long double const coupling) noexcept
    {
        _coupling = coupling;
    }

    /// Performs |ψ〉+= c * H|σ〉.
    auto operator()(long double const coeff, SpinVector const spin,
                    QuantumState& psi) const -> void
    {
        for (auto const& edge : edges()) {
            auto const aligned = spin[edge.first] == spin[edge.second];
            auto const sign    = static_cast<long double>(-1 + 2 * aligned);
            psi += {sign * coeff * coupling(), spin};
            if (!aligned) {
                psi += {2 * coeff * coupling(),
                        spin.flipped({edge.first, edge.second})};
            }
        }
    }
};

class Polynomial {
  private:
    Heisenberg               _h;
    std::vector<long double> _cs;
    QuantumState             _old;
    QuantumState             _current;

  public:
    Polynomial(Heisenberg hamiltonian, std::vector<long double> coeffs)
        : _h{std::move(hamiltonian)}, _cs{std::move(coeffs)}, _old{}, _current{}
    {}

    auto size() const noexcept { return _old.size(); }

    auto operator()(long double coeff, SpinVector const spin) -> Polynomial&
    {
        _old.clear();
        _old[spin] = coeff;
        for (auto const A : _cs) {
            // Performs current := (H - A) * old
            _current.clear();
            _current.reserve(_old.size());
            for (auto const& item : _old) {
                _current.emplace(item.first, -A * item.second);
            }
            for (auto const& item : _old) {
                _h(item.second, item.first, _current);
            }
            std::swap(_old, _current);
        }
        return *this;
    }

    auto print() const -> void
    {
        for (auto const& item : _old) {
            pybind11::print(static_cast<std::string>(item.first), item.second);
        }
    }

    template <class Float,
              class = std::enable_if_t<std::is_floating_point<Float>::value>>
    auto keys(pybind11::array_t<Float, pybind11::array::c_style> out) const
    {
        if (size() == 0) { return out; }
        auto       begin        = _old.begin();
        auto const number_spins = begin->first.size();
        auto       access       = out.template mutable_unchecked<2>();
        if (out.shape(0) != static_cast<pybind11::ssize_t>(size())
            || out.shape(1) != static_cast<pybind11::ssize_t>(number_spins)) {
            throw std::invalid_argument{
                "Output array has wrong shape: expected ("
                + std::to_string(size()) + ", " + std::to_string(number_spins)
                + "), but got (" + std::to_string(out.shape(0)) + ", "
                + std::to_string(out.shape(1)) + ")."};
        }
        for (auto i = size_t{0}; i < size(); ++i, ++begin) {
            begin->first.copy_to(access.mutable_data(i, 0));
        }
        return out;
    }

    template <class Float,
              class = std::enable_if_t<std::is_floating_point<Float>::value>>
    auto values(pybind11::array_t<Float> out) const -> pybind11::array_t<Float>
    {
        auto access = out.template mutable_unchecked<1>();
        if (out.shape(0) != static_cast<pybind11::ssize_t>(size())) {
            throw std::invalid_argument{
                "Output array has wrong shape: expected ("
                + std::to_string(size()) + ",), but got ("
                + std::to_string(out.shape(0)) + ",)."};
        }
        auto begin = _old.begin();
        for (auto i = size_t{0}; i < size(); ++i, ++begin) {
            access(i) = static_cast<Float>(begin->second);
        }
        return out;
    }
};

TCM_NAMESPACE_END
