#include "jacobian.hpp"
#include <c10/util/SmallVector.h>
#include <torch/autograd.h>

#include <iostream>

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
#endif
} // namespace detail

TCM_EXPORT auto jacobian(torch::jit::script::Module module, torch::Tensor in,
                         std::optional<torch::Tensor> out) -> torch::Tensor
{
    using std::begin, std::end;
    auto const parameters = [&module]() {
#if TCM_TORCH_VERSION >= 1004000 // >=1.4.0
        auto list = module.parameters();
        return torch::autograd::variable_list{begin(list), end(list)};
#else
        return detail::get_parameters(module);
#endif
    }();
    auto const number_parameters = std::accumulate(
        begin(parameters), end(parameters), int64_t{0},
        [](auto const acc, auto const& value) { return acc + value.numel(); });
    auto const batch_size = in.size(0);
    if (!out.has_value()) {
        out.emplace(in.new_empty({batch_size, number_parameters}));
    }
    auto forward = [method = module.get_method("forward")](auto x) mutable {
        return method({x}).toTensor();
    };
    auto results = forward(in);
    for (auto i = int64_t{0}; i < batch_size; ++i) {
        auto gradients = torch::autograd::grad(
            /*outputs=*/{results[i]}, /*inputs=*/parameters,
            /*grad_outputs=*/{}, /*retain_graph=*/true,
            /*create_graph=*/false, /*allow_unused=*/false);
        detail::flatten_cat_out(begin(gradients), end(gradients), (*out)[i]);
    }
    return *std::move(out);
}

TCM_NAMESPACE_END
