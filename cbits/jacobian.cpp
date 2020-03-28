#include "jacobian.hpp"
#include <c10/util/SmallVector.h>
#include <torch/autograd.h>

#include <omp.h>

TCM_NAMESPACE_BEGIN

namespace detail {
template <class Iterator>
auto flatten_cat_out(Iterator first, Iterator last, torch::Tensor out) -> void
{
    if (first != last) {
        using std::begin, std::end;
        c10::SmallVector<torch::Tensor, 32> flattened;
        std::transform(first, last, std::back_inserter(flattened),
                       [](auto const& variable) { return variable.flatten(); });
        torch::cat_out(out, c10::ArrayRef<torch::Tensor>{flattened});
    }
}

#if TCM_TORCH_VERSION < 1004000
// Older versions don't implement the recursion in get_parameters() function, so
// we have to do it ourselves.
namespace {
    auto get_parameters_impl(torch::autograd::variable_list&   parameters,
                             torch::jit::script::Module const& module,
                             bool recurse = true) -> void
    {
        for (auto const& p : module.get_parameters()) {
            parameters.push_back(p.value().toTensor());
        }
        if (recurse) {
            for (const auto& child : module.get_modules()) {
                get_parameters_impl(parameters, child, recurse);
            }
        }
    }

    auto get_parameters(torch::jit::script::Module const& module,
                        bool recurse = true) -> torch::autograd::variable_list
    {
        torch::autograd::variable_list parameters;
        parameters.reserve(32);
        get_parameters_impl(parameters, module, recurse);
        return parameters;
    }
} // namespace
#else
namespace {
    auto get_parameters(torch::jit::script::Module const& module,
                        bool recurse = true) -> torch::autograd::variable_list
    {
        torch::autograd::variable_list parameters;
        parameters.reserve(32);
        for (auto p : module.parameters(recurse)) {
            parameters.emplace_back(std::move(p));
        }
        return parameters;
    }
} // namespace
#endif

inline auto get_number_parameters(torch::autograd::variable_list const& parameters)
    -> int64_t
{
    return std::accumulate(
        begin(parameters), end(parameters), int64_t{0},
        [](auto const acc, auto const& value) { return acc + value.numel(); });
}

template <class Forward>
TCM_FORCEINLINE auto make_task(torch::Tensor const& in, Forward forward,
                               torch::autograd::variable_list const& parameters,
                               torch::Tensor const&                  out)
    -> std::function<auto(int64_t)->void>
{
    if (in.device().type() == c10::DeviceType::CUDA) {
        return [results = forward(in), &parameters, out](auto const i) {
            auto gradients = torch::autograd::grad(
                /*outputs=*/{results[i]}, /*inputs=*/parameters,
                /*grad_outputs=*/{}, /*retain_graph=*/true,
                /*create_graph=*/false, /*allow_unused=*/false);
            detail::flatten_cat_out(begin(gradients), end(gradients), out[i]);
        };
    }
    return
        [in, fn = std::move(forward), &parameters, out](auto const i) mutable {
            auto gradients = torch::autograd::grad(
                /*outputs=*/{fn(
                    torch::narrow(in, /*dim=*/0, /*start=*/i, /*length=*/1))},
                /*inputs=*/parameters,
                /*grad_outputs=*/{}, /*retain_graph=*/false,
                /*create_graph=*/false, /*allow_unused=*/false);
            detail::flatten_cat_out(begin(gradients), end(gradients), out[i]);
        };
}

} // namespace detail

TCM_EXPORT auto jacobian(torch::jit::script::Module const& module,
                         torch::Tensor in, std::optional<torch::Tensor> out,
                         int num_threads) -> torch::Tensor
{
    using std::begin, std::end;
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    auto const parameters = detail::get_parameters(module);
    auto const batch_size = in.size(0);
    if (!out.has_value()) {
        auto const dtype = parameters.at(0).scalar_type();
        auto const number_parameters =
            detail::get_number_parameters(parameters);
        out.emplace(torch::empty(
            {batch_size, number_parameters},
            torch::TensorOptions{}.device(in.device()).dtype(dtype)));
    }
    auto task = detail::make_task(
        in,
        [method = module.get_method("forward")](auto x) mutable {
            return method({x}).toTensor();
        },
        parameters, *out);
    if (num_threads > 1) {
        std::atomic_flag   error_flag    = ATOMIC_FLAG_INIT;
        std::exception_ptr exception_ptr = nullptr;
#pragma omp parallel for num_threads(num_threads) default(none)                \
    shared(task, error_flag, exception_ptr)
        for (auto i = int64_t{0}; i < batch_size; ++i) {
            try {
                task(i);
            }
            catch (...) {
                if (!error_flag.test_and_set()) {
                    exception_ptr = std::current_exception();
                }
            }
        }
        if (exception_ptr) { std::rethrow_exception(exception_ptr); }
    }
    else {
        for (auto i = int64_t{0}; i < batch_size; ++i) {
            task(i);
        }
    }
    return *std::move(out);
}

TCM_NAMESPACE_END
