#include "monte_carlo_v2.hpp"
#include "common.hpp"
#include "spin.hpp"

#include <torch/extension.h>
#include <torch/script.h>

TCM_NAMESPACE_BEGIN

_Options::_Options(unsigned const _number_spins, int const _magnetisation,
                   unsigned const _number_chains,
                   unsigned const _number_samples, unsigned const _sweep_size,
                   unsigned const _number_discarded)
    : number_spins{_number_spins}
    , magnetisation{_magnetisation}
    , number_chains{_number_chains}
    , number_samples{_number_samples}
    , sweep_size{_sweep_size}
    , number_discarded{_number_discarded}
{
    TCM_CHECK(0 < number_spins && number_spins < SpinVector::max_size(),
              std::invalid_argument,
              fmt::format("invalid number_spins: {}; expected a positive "
                          "integer not greater than {}",
                          number_spins, SpinVector::max_size()));
    TCM_CHECK(std::abs(magnetisation) <= number_spins
                  && (number_spins + magnetisation) % 2 == 0,
              std::invalid_argument,
              fmt::format("invalid magnetisation: {}", magnetisation));
    TCM_CHECK(
        0 < number_chains, std::invalid_argument,
        fmt::format("invalid number_chains: {}; expected a positive integer",
                    number_chains));
    TCM_CHECK(0 < sweep_size, std::invalid_argument,
              fmt::format("invalid sweep_size: {}; expected a positive integer",
                          sweep_size));
}

namespace {
class Kernel {
  private:
    gsl::not_null<RandomGenerator*> _generator;

    TCM_FORCEINLINE static auto
    magnetisation(gsl::span<SpinVector const> chunk) noexcept -> int
    {
        TCM_ASSERT(!chunk.empty(), "can't compute magnetisation, because there "
                                   "are not spin configurations in the batch");
        auto m = chunk[0].magnetisation();
        TCM_ASSERT(
            std::all_of(chunk.begin() + 1, chunk.end(),
                        [m](auto const& s) { return s.magnetisation() == m; }),
            "all spin configurations in the batch must have the same "
            "magnetisation");
        return m;
    }

    TCM_FORCEINLINE static auto size(gsl::span<SpinVector const> chunk,
                                     gsl::span<SpinVector const> other) noexcept
        -> unsigned
    {
        TCM_ASSERT(!chunk.empty(), "can't compute size, because there "
                                   "are not spin configurations in the batch");
        auto n = chunk[0].size();
        TCM_ASSERT(
            std::all_of(chunk.begin() + 1, chunk.end(),
                        [n](auto const& s) { return s.size() == n; }),
            "all spin configurations in the batch must have the same size");
        return n;
    }

  public:
    explicit constexpr Kernel(RandomGenerator& generator) noexcept
        : _generator{&generator}
    {}

    constexpr Kernel(Kernel const&) noexcept = default;
    constexpr Kernel(Kernel&&) noexcept      = default;
    constexpr Kernel& operator=(Kernel const&) noexcept = default;
    constexpr Kernel& operator=(Kernel&&) noexcept = default;

    auto operator()(gsl::span<SpinVector const> src,
                    gsl::span<SpinVector>       dst) const -> void
    {
        using std::begin;
        using std::end;
        TCM_ASSERT(src.size() == dst.size(), "dimensions don't match");
        auto m = magnetisation(src);
        auto n = static_cast<int>(size(src, dst));

        std::copy(begin(src), end(src), begin(dst));
        if (std::abs(m) < n) {
            std::for_each(begin(dst), end(dst), [this, n, m](auto& s) {
                auto const up =
                    s.find_nth_up(std::uniform_int_distribution<unsigned>{
                        0, static_cast<unsigned>(n + m) / 2 - 1}(*_generator));
                auto const down =
                    s.find_nth_down(std::uniform_int_distribution<unsigned>{
                        0, static_cast<unsigned>(n - m) / 2 - 1}(*_generator));
                s.flip(up);
                s.flip(down);
            });
        }
        TCM_ASSERT(
            std::all_of(begin(dst), end(dst),
                        [m](auto const& s) { return s.magnetisation() == m; }),
            "post-condition violated");
    }
};


template <class ForwardFn, class KernelFn>
class MarkovChain {
  public:
    // using ForwardFn = std::function<auto(torch::Tensor const&)->torch::Tensor>;
    // using KernelFn  = std::function<
    //     auto(gsl::span<SpinVector const>, gsl::span<SpinVector>)->void>;
    using SpinsT  = aligned_vector<SpinVector>;
    using ValuesT = aligned_vector<float>;

  private:
    ForwardFn const& _forward;
    KernelFn const&  _kernel;
    SpinsT           _current_x;
    SpinsT           _proposed_x;
    torch::Tensor    _proposed_x_unpacked;
    ValuesT          _current_y;
    RandomGenerator& _generator;
    size_t           _accepted;
    size_t           _count;

    static auto transition_probability(float current, float suggested) noexcept
        -> float
    {
        current *= current;
        suggested *= suggested;
        if (current <= suggested) return 1.0F;
        return std::pow(suggested / current, 0.25F);
    }

    auto random() -> float
    {
        return std::uniform_real_distribution<float>{0.0F, 1.0F}(_generator);
    }

    static auto check_initial_state(gsl::span<SpinVector const> chunk)
        -> std::tuple<unsigned, int>
    {
        using std::begin;
        using std::end;
        TCM_CHECK(!chunk.empty(), std::invalid_argument,
                  "initial state must not be empty");
        auto const n = chunk[0].size();
        auto const m = chunk[0].magnetisation();
        TCM_CHECK(
            std::all_of(begin(chunk) + 1, end(chunk),
                        [n](auto const& s) { return s.size() == n; }),
            std::invalid_argument,
            "initial state contains spin configurations of different lengths");
        TCM_CHECK(
            std::all_of(begin(chunk) + 1, end(chunk),
                        [m](auto const& s) { return s.magnetisation() == m; }),
            std::invalid_argument,
            "initial state contains spin configurations with different "
            "magnetisations");
        return std::make_tuple(n, m);
    }

  public:
    MarkovChain(ForwardFn forward, KernelFn kernel, SpinsT initial,
                RandomGenerator& generator)
        : _forward{forward}
        , _kernel{kernel}
        , _current_x{std::move(initial)}
        , _proposed_x{}
        , _proposed_x_unpacked{}
        , _current_y{}
        , _generator{generator}
        , _accepted{0}
        , _count{0}
    {
        unsigned n;
        int      m;
        std::tie(n, m) = check_initial_state(_current_x);
        _proposed_x.resize(_current_x.size());
        _proposed_x_unpacked = detail::make_tensor<float>(_current_x.size(), n);
        _current_y.resize(_current_x.size());

        // Forward propagation on the initial state
        unpack_to_tensor(begin(_current_x), end(_current_x),
                         _proposed_x_unpacked);
        torch::from_blob(_current_y.data(), {_current_y.size()}) =
            _forward(_proposed_x_unpacked);
    }

    auto reset_statistics() noexcept -> void
    {
        _accepted = 0;
        _count    = 0;
    }

    auto read() const noexcept
        -> std::tuple<gsl::span<SpinVector const>, gsl::span<float const>>
    {
        return std::make_tuple(gsl::span<SpinVector const>{_current_x},
                               gsl::span<float const>{_current_y});
    }

    auto acceptance() const noexcept -> float
    {
        if (_count == 0) { return std::numeric_limits<float>::quiet_NaN(); }
        return static_cast<float>(_accepted) / static_cast<float>(_count);
    }

    auto step() -> void
    {
        using std::begin;
        using std::end;

        _kernel(_current_x, _proposed_x);
        unpack_to_tensor(begin(_proposed_x), end(_proposed_x),
                         _proposed_x_unpacked);
        auto const output     = _forward(_proposed_x_unpacked);
        auto const proposed_y = output.template accessor<float, 1>();

        for (auto i = size_t{0}; i < _current_x.size(); ++i) {
            auto const p = transition_probability(
                _current_y[i], proposed_y[static_cast<int64_t>(i)]);
            if (random() <= p) {
                ++_accepted;
                _current_x[i] = _proposed_x[i];
                _current_y[i] = proposed_y[static_cast<int64_t>(i)];
            }
            ++_count;
        }
    }
};

template <class ForwardFn, class KernelFn>
auto make_markov_chain(
    ForwardFn const& forward, KernelFn const& kernel,
    typename MarkovChain<ForwardFn, KernelFn>::SpinsT initial,
    RandomGenerator& generator) -> MarkovChain<ForwardFn, KernelFn>
{
    return MarkovChain<ForwardFn, KernelFn>{forward, kernel,
                                            std::move(initial)};
}

template <class ForwardFn, class KernelFn>
auto make_markov_chain(ForwardFn const& forward, KernelFn const& kernel,
                       unsigned const number_chains,
                       unsigned const number_spins, int const magnetisation,
                       RandomGenerator& generator)
    -> MarkovChain<ForwardFn, KernelFn>
{
    typename MarkovChain<ForwardFn, KernelFn>::SpinsT initial;
    initial.reserve(number_chains);
    for (auto i = 0u; i < number_chains; ++i) {
        initial.push_back(
            SpinVector::random(number_spins, magnetisation, generator));
    }
    return MarkovChain<ForwardFn, KernelFn>{forward, kernel, std::move(initial),
                                            generator};
}


template <class ForwardFn>
auto _sample_some(ForwardFn const& psi, _Options const& options,
                  RandomGenerator* gen = nullptr)
{
    auto&      generator = (gen != nullptr) ? *gen : global_random_generator();
    auto       chain     = make_markov_chain(psi, Kernel{generator},
                                   options.number_chains, options.number_spins,
                                   options.magnetisation, generator);
    auto const count = (options.number_samples + options.number_chains - 1)
                       / options.number_chains;

    aligned_vector<SpinVector> spins(count * options.number_chains);
    aligned_vector<float>      values(count * options.number_chains);
    auto save = [&spins, &values, i = int64_t{0}](auto const& state) mutable {
        using std::begin;
        using std::end;
        auto const& xs = std::get<0>(state);
        auto const& ys = std::get<1>(state);
        std::copy(begin(xs), end(xs), begin(spins) + i);
        std::copy(begin(ys), end(ys), begin(values) + i);
        i += static_cast<int64_t>(xs.size());
    };

    for (auto i = 0u; i < options.number_discarded; ++i) {
        for (auto j = 0u; j < options.sweep_size; ++j) {
            chain.step();
        }
    }
    auto const thermalisation_acceptance = chain.acceptance();
    chain.reset_statistics();

    save(chain.read());
    for (auto i = 0u; i < count - 1; ++i) {
        for (auto j = 0u; j < options.sweep_size; ++j) {
            chain.step();
        }
        save(chain.read());
    }
    auto const acceptance = chain.acceptance();

    return std::make_tuple(std::move(spins), std::move(values), acceptance);
}

} // namespace

namespace v2 {
auto sample_some(std::string const& filename, _Options const& options)
    -> std::tuple<aligned_vector<SpinVector>, aligned_vector<float>, float>
{
    torch::NoGradGuard no_grad;
    auto               module = torch::jit::load(filename);
    auto               method = module.get_method("forward");
    return _sample_some(
        [&method](torch::Tensor const& x) {
            std::vector<torch::jit::IValue> stack{{x}};
            return method(std::move(stack)).toTensor();
        },
        options);
}

auto sample_some(std::function<auto(torch::Tensor const&)->torch::Tensor> state,
                 _Options const& options)
    -> std::tuple<aligned_vector<SpinVector>, aligned_vector<float>, float>
{
    torch::NoGradGuard no_grad;
    return _sample_some(state, options);
}
} // namespace v2

TCM_NAMESPACE_END
