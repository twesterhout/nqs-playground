#include "forward_propagator.hpp"
#include "heisenberg.hpp"

TCM_NAMESPACE_BEGIN

TCM_IMPORT auto apply(torch::Tensor spins, Heisenberg const& hamiltonian,
                      v2::ForwardT psi) -> torch::Tensor;

#if 0
    template <class State, class Hamiltonian> class PolynomialState {
        using _Polynomial = Polynomial<State, Hamiltonian>;

      public:
        using StateT       = State;
        using HamiltonianT = Hamiltonian;

      private:
        detail::Accumulator<State>   _accum;
        ForwardT<State>              _fn;
        std::shared_ptr<_Polynomial> _poly;

      public:
        PolynomialState(std::shared_ptr<_Polynomial> polynomial,
                        ForwardT<State> fn, unsigned batch_size);

        PolynomialState(PolynomialState const&)     = default;
        PolynomialState(PolynomialState&&) noexcept = default;
        auto operator=(PolynomialState const&) -> PolynomialState& = default;
        auto operator           =(PolynomialState&&) noexcept
            -> PolynomialState& = default;

        auto operator()(gsl::span<State const> spins) -> torch::Tensor;
    }; // }}}
#endif

TCM_NAMESPACE_END
