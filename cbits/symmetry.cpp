#include "symmetry.hpp"
#include "cpu/kernels.hpp"
// #include <boost/math/special_functions/cos_pi.hpp>
// #include <boost/math/special_functions/sin_pi.hpp>

// #include <iostream>

TCM_NAMESPACE_BEGIN

TCM_EXPORT SymmetryBase::SymmetryBase(unsigned const sector,
                                      unsigned const periodicity)
    : _sector{sector}, _periodicity{periodicity}, _eigenvalue{}
{
    TCM_CHECK(
        periodicity > 0, std::invalid_argument,
        fmt::format("invalid periodicity: {}; expected a positive integer",
                    periodicity));
    TCM_CHECK(sector < periodicity, std::invalid_argument,
              fmt::format("invalid sector: {}; expected an integer in [0, {})",
                          sector, periodicity));
    auto const arg =
        -static_cast<double>(2U * sector) / static_cast<double>(periodicity);
    _eigenvalue =
        std::complex<double>{std::cos(M_PI * arg), std::sin(M_PI * arg)};
}

namespace v2 {
Symmetry<64>::Symmetry(std::array<uint64_t, 6> const& forward,
                       std::array<uint64_t, 6> const& backward,
                       unsigned const sector, unsigned const periodicity)
    : SymmetryBase{sector, periodicity}, _fwd{forward}, _bwd{backward}
{}

auto Symmetry<64>::_internal_state() const noexcept -> _PickleStateT
{
    return std::make_tuple(sector(), periodicity(), _fwd, _bwd);
}

auto Symmetry<64>::_from_internal_state(_PickleStateT const& state)
    -> Symmetry<64>
{
    return Symmetry<64>{std::get<2>(state), std::get<3>(state),
                        std::get<0>(state), std::get<1>(state)};
}

Symmetry<512>::Symmetry(std::array<bits512, 9> const& forward,
                        std::array<bits512, 9> const& backward,
                        unsigned const sector, unsigned const periodicity)
    : SymmetryBase{sector, periodicity}, _fwd{forward}, _bwd{backward}
{}
} //namespace v2

#if 0
struct alignas(64) Symmetry8x64 {
    uint64_t             _fwds[6][8];
    uint64_t             _bwds[6][8];
    unsigned             _sectors[8];
    unsigned             _periodicities[8];
    std::complex<double> _eigenvalues[8];

    Symmetry8x64(gsl::span<v2::Symmetry<64> const> original);

    Symmetry8x64(Symmetry8x64 const&) noexcept = default;
    Symmetry8x64(Symmetry8x64&&) noexcept      = default;
    auto operator=(Symmetry8x64 const&) noexcept -> Symmetry8x64& = default;
    auto operator=(Symmetry8x64&&) noexcept -> Symmetry8x64& = default;
};
#endif

TCM_EXPORT
Symmetry8x64::Symmetry8x64(gsl::span<v2::Symmetry<64> const> original)
{
    TCM_CHECK(original.size() == 8, std::invalid_argument,
              fmt::format("expected a chunk of 8 symmetries, but got {}",
                          original.size()));
    for (auto i = 0U; i < 8U; ++i) {
        for (auto j = 0U; j < 6U; ++j) {
            _fwds[j][i] = original[i]._fwd[j];
            _bwds[j][i] = original[i]._bwd[j];
        }
        _sectors[i]       = original[i].sector();
        _periodicities[i] = original[i].periodicity();
        _eigenvalues[i]   = original[i].eigenvalue();
    }
}

TCM_EXPORT auto Symmetry8x64::operator()(uint64_t x, uint64_t out[8]) const
    noexcept -> void
{
    bfly(x, out, _fwds);
    ibfly(out, _bwds);
}

#if 0
auto full_info(gsl::span<v2::Symmetry<64> const> symmetries, uint64_t x)
    -> std::tuple</*representative=*/uint64_t,
                  /*eigenvalue=*/std::complex<double>, /*norm=*/double>
{
    if (symmetries.empty()) {
        return std::make_tuple(x, std::complex<double>{1.0, 0.0}, 1.0);
    }
    auto const count = static_cast<unsigned>(symmetries.size());
    auto       repr  = x;
    auto       phase = 0.0;
    auto       norm  = 0.0;
    for (auto const& symmetry : symmetries) {
        auto const y = symmetry(x);
        if (y < repr) {
            repr  = y;
            phase = symmetry.phase();
        }
        else if (y == x) {
            // We're actually interested in
            // std::conj(first->eigenvalue()).real(), but Re[z*] == Re[z].
            norm += symmetry.eigenvalue().real();
        }
    }

    // We need to detect the case when norm is not zero, but only because of
    // inaccurate arithmetics
    constexpr auto epsilon = 1.0e-5;
    if (std::abs(norm) <= epsilon) { norm = 0.0; }
    TCM_CHECK(
        norm >= 0.0, std::runtime_error,
        fmt::format("state {} appears to have negative squared norm {} :/", x,
                    norm));
    norm = std::sqrt(norm / static_cast<double>(count));

    // #if defined(TCM_DEBUG) // This is a sanity check
    constexpr auto almost_equal = [](double const a, double const b) {
        using std::max, std::abs;
        constexpr auto const rtol = 1.0e-7;
        return abs(b - a) <= max(abs(a), abs(b)) * rtol;
    };
    if (norm > 0.0) {
        for (auto const& symmetry : symmetries) {
            auto const y = symmetry(x);
            if (y == repr) {
                TCM_CHECK(
                    almost_equal(symmetry.phase(), phase), std::logic_error,
                    fmt::format("The result of a long discussion that gσ "
                                "= hσ => λ(g) = λ(h) is wrong: {} != {}, σ={}",
                                symmetry.phase(), phase, y));
            }
        }
    }
    // #endif
    auto const arg = 2.0 * M_PI * phase;
    return std::make_tuple(
        repr, std::complex<double>{std::cos(arg), std::sin(arg)}, norm);
}
#endif

TCM_EXPORT auto representative(gsl::span<Symmetry8x64 const>     symmetries,
                               gsl::span<v2::Symmetry<64> const> other,
                               uint64_t const                    x) noexcept
    -> std::tuple<uint64_t, double, ptrdiff_t>
{
    alignas(32) uint64_t buffer[8];

    auto repr = x;
    auto norm = 0.0;
    auto g    = ptrdiff_t{0};
    for (auto i = uint64_t{0}; i < symmetries.size(); ++i) {
        symmetries[i](x, buffer);
        for (auto n = 0; n < 8; ++n) {
            if (buffer[n] < repr) {
                repr = buffer[n];
                g    = static_cast<ptrdiff_t>(i) + n;
            }
            else if (buffer[n] == x) {
                norm += symmetries[i]._eigenvalues[n].real();
            }
        }
    }
    for (auto i = uint64_t{0}; i < other.size(); ++i) {
        auto const y = other[i](x);
        if (y < repr) {
            repr = y;
            g    = static_cast<ptrdiff_t>(8U * symmetries.size() + i);
        }
        else if (y == x) {
            norm += other[i].eigenvalue().real();
        }
    }

    // We need to detect the case when norm is not zero, but only because of
    // inaccurate arithmetics
    constexpr auto norm_threshold = 1.0e-5;
    if (std::abs(norm) <= norm_threshold) { norm = 0.0; }
    TCM_ASSERT(norm >= 0, "");
    norm = std::sqrt(
        norm / static_cast<double>(8U * symmetries.size() + other.size()));
    return {repr, norm, g};
}

template <bool Representative, bool Eigenvalue, bool Norm, class Symmetry>
TCM_FORCEINLINE auto
_kernel(Symmetry const& symmetry, typename Symmetry::StateT const& x,
        typename Symmetry::StateT& repr, double& phase, double& norm) -> void
{
    auto const y = symmetry(x);
    if constexpr (Representative || Eigenvalue) {
        if (y < repr) {
            if constexpr (Representative) { repr = y; }
            if constexpr (Eigenvalue) { phase = symmetry.phase(); }
            return;
        }
    }
    if constexpr (Norm) {
        if (y == x) { norm += symmetry.eigenvalue().real(); }
    }
}

// Returns a single element tuple when Norm is true and an empty tuple otherwise
template <bool UseValue, class T>
TCM_FORCEINLINE auto _to_tuple_if(T const& value)
{
    if constexpr (UseValue) { return std::tuple<T>{value}; }
    else {
        return std::tuple{};
    }
}

template <bool Eigenvalue>
TCM_FORCEINLINE auto _make_eigenvalue(double const norm, double const phase)
{
    if constexpr (Eigenvalue) {
        if (norm > 0.0) {
            auto const arg = 2.0 * M_PI * phase;
            return std::make_tuple(
                std::complex<double>{std::cos(arg), std::sin(arg)});
        }
        constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(std::complex<double>{NaN, NaN});
    }
    else {
        return std::tuple{};
    }
}

template <bool Representative = true, bool Eigenvalue = true, bool Norm = true,
          class Symmetry = v2::Symmetry<512>>
auto _full_info(gsl::span<Symmetry const>        symmetries,
                typename Symmetry::StateT const& x)
{
    if (symmetries.empty()) {
        return std::tuple_cat(
            _to_tuple_if<Representative>(x),
            _to_tuple_if<Eigenvalue>(std::complex<double>{1.0, 0.0}),
            _to_tuple_if<Norm>(1.0));
    }
    auto repr  = x;
    auto phase = 0.0;
    auto norm  = 0.0;
    for (auto const& symmetry : symmetries) {
        _kernel<Representative, Eigenvalue, Norm, Symmetry>(symmetry, x, repr,
                                                            phase, norm);
#if 0
        auto const y = symmetry(x);
        if (y < repr) {
            repr  = y;
            phase = symmetry.phase();
        }
        else if (y == x) {
            // We're actually interested in
            // std::conj(first->eigenvalue()).real(), but Re[z*] == Re[z].
            norm += symmetry.eigenvalue().real();
        }
#endif
    }

    if constexpr (Norm) {
        // We need to detect the case when norm is not zero, but only because of
        // inaccurate arithmetics
        constexpr auto norm_threshold = 1.0e-5;
        if (std::abs(norm) <= norm_threshold) { norm = 0.0; }
        TCM_CHECK(
            norm >= 0.0, std::runtime_error,
            fmt::format("state {} appears to have negative squared norm {} :/",
                        x, norm));
        norm = std::sqrt(norm / static_cast<double>(symmetries.size()));
    }

#if defined(TCM_DEBUG)
    constexpr auto almost_equal = [](double const a, double const b) {
        using std::max, std::abs;
        constexpr auto const rtol = 1.0e-7;
        return abs(b - a) <= max(abs(a), abs(b)) * rtol;
    };
    if (Norm && norm > 0.0) {
        for (auto const& symmetry : symmetries) {
            auto const y = symmetry(x);
            if (y == repr) {
                TCM_CHECK(
                    almost_equal(symmetry.phase(), phase), std::logic_error,
                    fmt::format("The result of a long discussion that gσ "
                                "= hσ => λ(g) = λ(h) is wrong: {} != {}, σ={}",
                                symmetry.phase(), phase, y));
            }
        }
    }
#endif

    return std::tuple_cat(_to_tuple_if<Representative>(repr),
                          _make_eigenvalue<Eigenvalue>(norm, phase),
                          _to_tuple_if<Norm>(norm));
#if 0
    if (norm > 0.0) {
        auto const arg = 2.0 * M_PI * phase;
        return std::make_tuple(
            repr, std::complex<double>{std::cos(arg), std::sin(arg)}, norm);
    }
    else {
        constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
        return std::make_tuple(repr, std::complex<double>{NaN, NaN}, norm);
    }
#endif
}

TCM_EXPORT auto full_info(gsl::span<v2::Symmetry<64> const> symmetries,
                          uint64_t                          x)
    -> std::tuple</*representative=*/uint64_t,
                  /*eigenvalue=*/std::complex<double>, /*norm=*/double>
{
    return _full_info<true, true, true>(symmetries, x);
}

TCM_EXPORT auto full_info(gsl::span<v2::Symmetry<512> const> symmetries,
                          bits512 const&                     x)
    -> std::tuple</*representative=*/bits512,
                  /*eigenvalue=*/std::complex<double>, /*norm=*/double>
{
    return _full_info<true, true, true>(symmetries, x);
}

TCM_NAMESPACE_END
