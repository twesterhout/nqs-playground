#include "symmetry.hpp"
#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/special_functions/sin_pi.hpp>

#include <iostream>

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
    _eigenvalue = std::complex<double>{boost::math::cos_pi(arg),
                                       boost::math::sin_pi(arg)};
}

namespace v2 {
Symmetry<64>::Symmetry(std::array<uint64_t, 6> const& forward,
                       std::array<uint64_t, 6> const& backward,
                       unsigned const sector, unsigned const periodicity)
    : SymmetryBase{sector, periodicity}, _fwd{forward}, _bwd{backward}
{}

Symmetry<512>::Symmetry(std::array<bits512, 9> const& forward,
                        std::array<bits512, 9> const& backward,
                        unsigned const sector, unsigned const periodicity)
    : SymmetryBase{sector, periodicity}, _fwd{forward}, _bwd{backward}
{}
} //namespace v2

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
