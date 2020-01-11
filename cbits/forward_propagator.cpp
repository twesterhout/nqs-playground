#include "forward_propagator.hpp"
#include <torch/jit.h>

#include <iostream>

TCM_NAMESPACE_BEGIN

namespace detail {

TaskBuilder::TaskBuilder(v2::ForwardT psi, uint64_t batch_size)
    : _i{0}, _spins_ptr{nullptr}, _coeffs_ptr{nullptr}, _batch_size{batch_size}
{
    TCM_CHECK(batch_size > 0, std::invalid_argument,
              fmt::format("invalid batch_size: {}; expected a positive integer",
                          batch_size));
    prepare(std::move(psi));
}

auto TaskBuilder::prepare(v2::ForwardT fn) -> void
{
    auto spins  = torch::empty({static_cast<int64_t>(_batch_size), 1L},
                              torch::TensorOptions{}.dtype(torch::kInt64));
    auto coeffs = torch::empty({static_cast<int64_t>(_batch_size), 2L},
                               torch::TensorOptions{}.dtype(torch::kFloat32));
    _next_task  = Task{/*psi=*/std::move(fn), /*spins=*/std::move(spins),
                      /*coeffs=*/std::move(coeffs),
                      /*counts=*/std::vector<uint64_t>{}, /*complete=*/true};
    _i          = 0;
    _spins_ptr  = static_cast<uint64_t*>(_next_task.spins.data_ptr());
    _coeffs_ptr =
        static_cast<std::complex<float>*>(_next_task.coeffs.data_ptr());
}

auto TaskBuilder::add_junk() -> void
{
    TCM_ASSERT(!empty(), "precondition violated");
    _next_task.counts.push_back(0);
    _next_task.complete = true;

    auto spin = _spins_ptr[_i - 1];
    for (; _i < _batch_size; ++_i) {
        _spins_ptr[_i]  = spin;
        _coeffs_ptr[_i] = std::complex<float>{0.0f, 0.0f};
    }
    TCM_ASSERT(full(), "postcondition violated");
}

auto TaskBuilder::submit(bool prepare_next) -> Task
{
    TCM_ASSERT(full(), "buffer is not full yet");
    auto task = std::move(_next_task);
    if (prepare_next) { prepare(task.psi); }
    return task;
}

namespace {
    auto dotu(torch::Tensor a, torch::Tensor b) -> std::complex<float>
    {
        static auto p = [] {
            auto  cu = torch::jit::compile(R"JIT(
            def dotu(a, b):
                real_a = a[:, 0]
                imag_a = a[:, 1]
                real_b = b[:, 0]
                imag_b = b[:, 1]
                return torch.sum(real_a * real_b - imag_a * imag_b), \
                    torch.sum(real_a * imag_b + imag_a * real_b)
        )JIT");
            auto& fn = cu->get_function("dotu");
            return torch::jit::StrongFunctionPtr{std::move(cu),
                                                 std::addressof(fn)};
        }();
        auto out = std::move(
            (*p.function_)({std::move(a), std::move(b)}).toTuple()->elements());
        return std::complex<float>{std::move(out[0]).toTensor().item<float>(),
                                   std::move(out[1]).toTensor().item<float>()};
    }
} // namespace

auto TaskBuilder::Task::operator()() const
    -> std::tuple<float, bool, std::vector<std::complex<float>>>
{
    torch::NoGradGuard no_grad;

    std::cerr << "calling this->psi(spins)...\n";
    auto output = this->psi(spins);
    std::cerr << "received output\n";
    auto const batch_size = this->spins.size(0);
    TCM_CHECK_SHAPE("output tensor", output, {batch_size, 2});
    TCM_CHECK_CONTIGUOUS("output tensor", output);
    auto real = torch::narrow(output, /*dim=*/1, /*start=*/0, /*length=*/1);
    auto imag = torch::narrow(output, /*dim=*/1, /*start=*/1, /*length=*/1);
    std::cerr << "constructed real & imag\n";

    auto const scale = torch::max(real).item<float>();
    TCM_CHECK(!std::isnan(scale), std::runtime_error,
              "NaN encountered in neural network output");
    std::cerr << "calculated scale\n";

    // The following computes complex-valued exp of (output - scale) in-place
    real -= scale;
    real.exp_();
    auto cos = torch::cos(imag);
    torch::sin_(imag);
    imag *= real;
    real *= cos;
    std::cerr << "updated output\n";

    std::vector<std::complex<float>> results;
    results.reserve(this->counts.size());

    TCM_ASSERT(!counts.empty(), "");
    auto offset = int64_t{0};
    auto j      = size_t{0};
    for (; j < this->counts.size() - 1; offset += this->counts[j++]) {
        auto r = dotu(torch::narrow(coeffs, /*dim=*/0,
                                    /*start=*/offset,
                                    /*length=*/counts[j]),
                      torch::narrow(output, /*dim=*/0, /*start=*/offset,
                                    /*length=*/counts[j]));
        results.push_back(r);
    }
    if (!(counts[j] == 0 && offset != batch_size)) {
        auto r = dotu(torch::narrow(coeffs, /*dim=*/0,
                                    /*start=*/offset,
                                    /*length=*/counts[j]),
                      torch::narrow(output, /*dim=*/0, /*start=*/offset,
                                    /*length=*/counts[j]));
        results.push_back(r);
    }

    std::cerr << "returning results\n";
    return {scale, complete, std::move(results)};
}

Accumulator::Accumulator(v2::ForwardT psi, gsl::span<std::complex<float>> out,
                         unsigned batch_size)
    : _builder{std::move(psi), batch_size}, _store{out}, _state{}, _futures{}
{}

auto Accumulator::reset(gsl::span<std::complex<float>> out) -> void
{
    TCM_CHECK(_builder.empty(), std::runtime_error,
              fmt::format("precondition violated"));
    TCM_CHECK(_futures.empty(), std::runtime_error,
              fmt::format("precondition violated"));
    _store = output_type{out};
    _state = state_type{};
}

auto Accumulator::drain(unsigned const count) -> void
{
    TCM_ASSERT(count <= _futures.size(), "");
    for (auto i = 0U; i < count; ++i) {
        auto future = std::move(_futures.front());
        _futures.pop();
        std::cerr << "calling future::get()...\n";
        auto value = future.get();
        std::cerr << "calling process...\n";
        process(value);
    }
}

auto Accumulator::process(result_type result) -> void
{
    auto const [scale, complete, values] = std::move(result);
    if (!values.empty()) {
        auto i = size_t{0};
        _state += state_type{values[i], scale};
        if (values.size() > 1) {
            _store(_state.get_log());
            for (++i; i < values.size() - 1; ++i) {
                _store(state_type{values[i], scale}.get_log());
            }
            _state = state_type{values[i], scale};
            if (complete) {
                _store(_state.get_log());
                _state = state_type{};
            }
        }
        else {
            if (complete) {
                _store(_state.get_log());
                _state = state_type{};
            }
        }
    }
}

auto Accumulator::finalize() -> void
{
    TCM_ASSERT(!_builder.full(), "precondition violated");
    if (!_builder.empty()) {
        std::cerr << "adding junk...\n";
        _builder.add_junk();
        drain_if_needed();
        std::cerr << "draining done\n";
        _futures.push(std::async(_builder.submit()));
        std::cerr << "submitted a task\n";
    }
    TCM_ASSERT(_builder.empty(), "postcondition violated");
    drain(_futures.size());
}

} // namespace detail

TCM_NAMESPACE_END
