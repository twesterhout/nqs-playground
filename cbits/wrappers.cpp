// Copyright (c) 2020, Tom Westerhout
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

#include "wrappers.hpp"

TCM_NAMESPACE_BEGIN

TCM_EXPORT auto check_status_code(ls_error_code const code) -> void
{
    if (TCM_UNLIKELY(code != LS_SUCCESS)) {
        auto deleter = [](auto const* s) { ls_destroy_string(s); };
        auto c_str =
            std::unique_ptr<char const, decltype(deleter)>{ls_error_to_string(code), deleter};
        TCM_ERROR(
            std::runtime_error,
            fmt::format("lattice_symmetries failed with error code {}: {}", code, c_str.get()));
    }
}

TCM_EXPORT auto view_as_operator(ls_operator const& op) -> std::tuple<OperatorT, uint64_t>
{
    auto max_required_size = ls_operator_max_buffer_size(&op);
    auto action = [&op, max_required_size](bits512 const& spin, std::complex<double> coeff,
                                           gsl::span<bits512>              out_spins,
                                           gsl::span<std::complex<double>> out_coeffs) {
        TCM_CHECK(out_spins.size() >= max_required_size, std::runtime_error,
                  fmt::format("out_spins buffer is too short: {}; expected at least {}",
                              out_spins.size(), max_required_size));
        TCM_CHECK(out_coeffs.size() >= max_required_size, std::runtime_error,
                  fmt::format("out_coeffs buffer is too short: {}; expected at least {}",
                              out_coeffs.size(), max_required_size));
        struct cxt_t {
            ls_bits512*           spins_ptr;
            std::complex<double>* coeffs_ptr;
            uint64_t              offset;
            uint64_t              max_size;
        } cxt{reinterpret_cast<ls_bits512*>(out_spins.data()), out_coeffs.data(), 0,
              max_required_size}; // FIXME: move to ls_bits512 everywhere
        auto callback = [](ls_bits512 const* _spin, void const* _coeff, void* cxt_raw) {
            auto* _cxt = static_cast<cxt_t*>(cxt_raw);
            TCM_CHECK(_cxt->offset < _cxt->max_size, std::runtime_error, "");
            _cxt->spins_ptr[_cxt->offset]  = *_spin;
            _cxt->coeffs_ptr[_cxt->offset] = *static_cast<std::complex<double> const*>(_coeff);
            ++(_cxt->offset);
            return LS_SUCCESS;
        };
        check_status_code(
            ls_operator_apply(&op, reinterpret_cast<ls_bits512 const*>(&spin), callback, &cxt));
        TCM_CHECK(cxt.offset <= max_required_size, std::runtime_error, "buffer overflow");
        return cxt.offset;
    };
    return {std::move(action), max_required_size};
}

TCM_EXPORT auto random_spin(ls_spin_basis const& basis, RandomGenerator& generator) -> ls_bits512
{
    auto const number_spins = ls_get_number_spins(&basis);
    ls_bits512 spin;
    set_zero(spin);
    if (ls_get_hamming_weight(&basis) == -1) {
        auto i = 0U;
        for (; i < number_spins / 64U; ++i) {
            spin.words[i] = std::uniform_int_distribution<uint64_t>{}(generator);
        }
        auto const rest = number_spins % 64U;
        if (rest != 0) {
            spin.words[i] =
                std::uniform_int_distribution<uint64_t>{0, (uint64_t{1} << rest) - 1}(generator);
        }
    }
    else {
        auto const           m = static_cast<unsigned>(ls_get_hamming_weight(&basis));
        std::vector<uint8_t> buffer(number_spins);
        std::fill(std::begin(buffer), std::next(std::begin(buffer), m), uint8_t{1});
        std::fill(std::next(std::begin(buffer), m), std::end(buffer), uint8_t{0});
        std::shuffle(std::begin(buffer), std::end(buffer), generator);
        auto word = 0U;
        auto i    = 0U;
        for (auto const b : buffer) {
            if (b) { spin.words[word] |= static_cast<uint64_t>(b) << i; }
            ++i;
            if (i == 64) {
                i = 0U;
                ++word;
            }
        }
    }
    ls_bits512 repr;
    set_zero(repr);
    std::complex<double> character;
    double               norm;
    ls_get_state_info(&basis, &spin, &repr, &character, &norm);
    if (norm > 0.0) { return repr; }
    return random_spin(basis, generator);
}

TCM_NAMESPACE_END
