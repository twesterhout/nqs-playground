#include "tensor_info.hpp"
#include "bits512.hpp"
#include "errors.hpp"
#include <boost/align/is_aligned.hpp>

TCM_NAMESPACE_BEGIN

namespace detail {
TCM_FORCEINLINE auto packed_spins_checks(torch::Tensor x, char const* name)
    -> void
{
    auto const arg_name = name != nullptr ? name : "tensor";
    auto const sizes    = x.sizes();
    auto const strides  = x.strides();
    TCM_CHECK(
        sizes.size() == 2 && sizes[1] == 8, std::invalid_argument,
        fmt::format(
            "{} has wrong shape: [{}]; packed spins must have shape [?, 8]",
            arg_name, fmt::join(sizes, ",")));
    TCM_CHECK(strides[1] == 1, std::invalid_argument,
              fmt::format("{} has wrong strides: [{}]; packed spins must be "
                          "contiguous along the second dimension",
                          arg_name, fmt::join(strides, ",")));
    TCM_CHECK_TYPE(arg_name, x, torch::kInt64);
}

TCM_FORCEINLINE auto complex_checks(torch::Tensor x, char const* name)
{
    auto const arg_name = name != nullptr ? name : "tensor";
    auto const sizes    = x.sizes();
    auto const strides  = x.strides();
    TCM_CHECK(
        sizes.size() == 2 && sizes[1] == 2, std::invalid_argument,
        fmt::format(
            "{} has wrong shape: [{}]; complex tensor must have shape [?, 2]",
            arg_name, fmt::join(sizes, ",")));
    TCM_CHECK(strides[1] == 1, std::invalid_argument,
              fmt::format("{} has wrong strides: [{}]; complex tensor must be "
                          "contiguous along the second dimension",
                          arg_name, fmt::join(strides, ",")));
    TCM_CHECK_TYPE(arg_name, x, torch::kFloat32);
}
} // namespace detail

template <>
TCM_EXPORT auto obtain_tensor_info<bits512, false>(torch::Tensor x,
                                                   char const*   name)
    -> TensorInfo<bits512>
{
    auto const arg_name = name != nullptr ? name : "tensor";
    auto*      data     = x.data_ptr();
    TCM_CHECK(boost::alignment::is_aligned(64U, data), std::invalid_argument,
              fmt::format("{} must be aligned to 64-byte boundary", arg_name));
    auto const size   = x.sizes()[0];
    auto const stride = [&x, &arg_name]() {
        auto const delta = x.strides()[0];
        TCM_CHECK(delta % 8L == 0, std::runtime_error,
                  fmt::format("{}'s stride along the first dimension must be a "
                              "multiple of 8; got {}",
                              arg_name, delta));
        // Because bits512 consists of 8 int64_t
        return delta / 8L;
    }();
    return TensorInfo<bits512>{static_cast<bits512*>(data), &size, &stride};
}

template <>
TCM_EXPORT auto obtain_tensor_info<bits512, true>(torch::Tensor x,
                                                  char const*   name)
    -> TensorInfo<bits512>
{
    detail::packed_spins_checks(x, name);
    return obtain_tensor_info<bits512, false>(std::move(x), name);
}

template <>
TCM_EXPORT auto obtain_tensor_info<std::complex<float>, false>(torch::Tensor x,
                                                               char const* name)
    -> TensorInfo<std::complex<float>>
{
    auto const arg_name = name != nullptr ? name : "tensor";
    auto*      data     = x.data_ptr();
    auto const size     = x.sizes()[0];
    auto const stride   = [&x, &arg_name]() {
        auto const delta = x.strides()[0];
        TCM_CHECK(delta % 2L == 0, std::runtime_error,
                  fmt::format("{}'s stride along the first dimension must be a "
                              "multiple of 2; got {}",
                              arg_name, delta));
        // Because complex<float> consists of 2 floats
        return delta / 2L;
    }();
    return TensorInfo<std::complex<float>>{
        static_cast<std::complex<float>*>(data), &size, &stride};
}

template <>
TCM_EXPORT auto obtain_tensor_info<std::complex<float>, true>(torch::Tensor x,
                                                              char const* name)
    -> TensorInfo<std::complex<float>>
{
    detail::complex_checks(x, name);
    return obtain_tensor_info<std::complex<float>, false>(std::move(x), name);
}

template <>
TCM_EXPORT auto obtain_tensor_info<uint64_t, false>(torch::Tensor x,
                                                    char const*   name)
    -> TensorInfo<uint64_t>
{
    return TensorInfo<uint64_t>{static_cast<uint64_t*>(x.data_ptr()),
                                x.sizes().data(), x.strides().data()};
}

template <>
TCM_EXPORT auto obtain_tensor_info<uint64_t, true>(torch::Tensor x,
                                                   char const*   name)
    -> TensorInfo<uint64_t>
{
    auto const arg_name = name != nullptr ? name : "tensor";
    auto const sizes    = x.sizes();
    TCM_CHECK(sizes.size() == 1, std::invalid_argument,
              fmt::format(
                  "{} has wrong shape: [{}]; expected a one-dimensional tensor",
                  arg_name, fmt::join(sizes, ",")));
    TCM_CHECK_TYPE(arg_name, x, torch::kInt64);
    return obtain_tensor_info<uint64_t, false>(x, name);
}

#define TRIVIAL_IMPLEMENTATION(type, torch_type)                               \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type, false>(torch::Tensor x,           \
                                                    char const*   name)        \
        ->TensorInfo<type>                                                     \
    {                                                                          \
        return TensorInfo<type>{static_cast<type*>(x.data_ptr()),              \
                                x.sizes().data(), x.strides().data()};         \
    }                                                                          \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type, true>(torch::Tensor x,            \
                                                   char const*   name)         \
        ->TensorInfo<type>                                                     \
    {                                                                          \
        auto const arg_name = name != nullptr ? name : "tensor";               \
        auto const sizes    = x.sizes();                                       \
        TCM_CHECK(                                                             \
            sizes.size() == 1, std::invalid_argument,                          \
            fmt::format(                                                       \
                "{} has wrong shape: [{}]; expected a one-dimensional tensor", \
                arg_name, fmt::join(sizes, ",")));                             \
        TCM_CHECK_TYPE(arg_name, x, torch_type);                               \
        return obtain_tensor_info<type, false>(x, name);                       \
    }

#define ADD_CONST_OVERLOADS(type)                                              \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type const, true>(torch::Tensor x,      \
                                                         char const*   name)   \
        ->TensorInfo<type const>                                               \
    {                                                                          \
        return obtain_tensor_info<type, true>(std::move(x), name);             \
    }                                                                          \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type const, false>(torch::Tensor x,     \
                                                          char const*   name)  \
        ->TensorInfo<type const>                                               \
    {                                                                          \
        return obtain_tensor_info<type, false>(std::move(x), name);            \
    }

TRIVIAL_IMPLEMENTATION(float, torch::kFloat32)
TRIVIAL_IMPLEMENTATION(double, torch::kFloat64)
TRIVIAL_IMPLEMENTATION(int64_t, torch::kInt64)

ADD_CONST_OVERLOADS(float)
ADD_CONST_OVERLOADS(double)
ADD_CONST_OVERLOADS(uint64_t)
ADD_CONST_OVERLOADS(bits512)
ADD_CONST_OVERLOADS(std::complex<float>)

#undef TRIVIAL_IMPLEMENTATION
#undef ADD_CONST_OVERLOADS

TCM_NAMESPACE_END
