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

#include "simple_polynomial.hpp"
#include "errors.hpp"

TCM_NAMESPACE_BEGIN

struct Polynomial::Buffer {
  private:
    bits512*      _spins;
    complex_type* _coeffs;
    uint64_t      _size;
    uint64_t      _max_size;

    constexpr auto check_state() const noexcept -> void
    {
        TCM_ASSERT(_spins != nullptr && _coeffs != nullptr,
                   "precondition violated");
        TCM_ASSERT(_size <= _max_size, "precondition violated");
    }

  public:
    Buffer(gsl::span<bits512> const spins, gsl::span<complex_type> const coeffs)
        : _spins{spins.data()}
        , _coeffs{coeffs.data()}
        , _size{0}
        , _max_size{spins.size()}
    {
        TCM_CHECK(_spins != nullptr, std::invalid_argument, "spins is empty");
        TCM_CHECK(_max_size > 0, std::invalid_argument, "spins is empty");
        TCM_CHECK(_coeffs != nullptr, std::invalid_argument, "coeffs is empty");
        TCM_CHECK(
            spins.size() == coeffs.size(), std::invalid_argument,
            fmt::format("spins and coeffs have different lengths: {} != {}",
                        spins.size(), coeffs.size()));
    }

    constexpr auto source() const noexcept
        -> std::tuple<gsl::span<bits512>, gsl::span<complex_type>>
    {
        check_state();
        return {{_spins, _size}, {_coeffs, _size}};
    }

    constexpr auto source_size() const noexcept { return _size; }

    constexpr auto destination() const noexcept
        -> std::tuple<gsl::span<bits512>, gsl::span<complex_type>>
    {
        check_state();
        return {{_spins + _size, _max_size - _size},
                {_coeffs + _size, _max_size - _size}};
    }

    constexpr auto destination_size() const noexcept
    {
        check_state();
        return _max_size - _size;
    }

    constexpr auto update_after_write(uint64_t const written) -> void
    {
        check_state();
        TCM_ASSERT(written <= _max_size - _size, "buffer overflow");
        _size += written;
    }
};

TCM_EXPORT
Polynomial::Polynomial(OperatorT hamiltonian, std::vector<complex_type> roots,
                       bool const normalising, uint64_t const max_states)
    : _hamiltonian{std::move(hamiltonian)}
    , _roots{std::move(roots)}
    , _max_states{max_states}
    , _normalising{normalising}
{
    TCM_CHECK(_hamiltonian, std::invalid_argument,
              "hamiltonian must not be None");
    TCM_CHECK(!_roots.empty(), std::invalid_argument,
              "zero-degree polynomials are not supported");
    TCM_CHECK(_max_states > 0, std::invalid_argument,
              "max_states must be positive");
}

TCM_EXPORT auto Polynomial::degree() const noexcept -> uint64_t
{
    return _roots.size();
}

TCM_EXPORT auto Polynomial::max_states() const noexcept -> uint64_t
{
    return _max_states;
}

namespace detail {
inline auto normalise(gsl::span<complex_type> xs) -> void
{
    auto norm = 0.0;
    for (auto const& y : xs) {
        norm += std::norm(y);
    }
    auto const scale = 1.0 / std::sqrt(norm);
    for (auto& y : xs) {
        y *= scale;
    }
}
} // namespace detail

TCM_EXPORT auto Polynomial::operator()(bits512 const& spin, complex_type coeff,
                                       gsl::span<bits512>      out_spins,
                                       gsl::span<complex_type> out_coeffs) const
    -> uint64_t
{
    TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
              std::runtime_error,
              fmt::format("invalid coefficient {} + {}j; expected a finite "
                          "(i.e. either normal, subnormal or zero)",
                          coeff.real(), coeff.imag()));
    Buffer buffer{out_spins, out_coeffs};
    TCM_CHECK(buffer.destination_size() > 0, std::runtime_error,
              fmt::format("output buffer is of insufficient size: {}",
                          buffer.destination_size()));
    out_spins[0]  = spin;
    out_coeffs[0] = coeff;
    buffer.update_after_write(1);
    for (auto const& root : _roots) {
        iteration(root, buffer);
        if (_normalising) { detail::normalise(std::get<1>(buffer.source())); }
    }
    return buffer.source_size();
}

TCM_EXPORT auto Polynomial::iteration(complex_type root, Buffer& buffer) const
    -> void
{
    auto const [spins, coeffs] = buffer.source();
    auto const init_size       = buffer.source_size();
    for (auto i = uint64_t{0}; i < init_size; ++i) {
        auto const [out_spins, out_coeffs] = buffer.destination();
        auto written = _hamiltonian(spins[i], coeffs[i], out_spins, out_coeffs);
        coeffs[i] *= -root;
        // This is a small optimisation which uses the fact that Heisenberg and
        // Operator output diagonal values last
        if (written > 0 && out_spins[written - 1] == spins[i]) {
            coeffs[i] += out_coeffs[written - 1];
            --written;
        }
        buffer.update_after_write(written);
    }
}

TCM_NAMESPACE_END
