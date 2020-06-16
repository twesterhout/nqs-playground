#include "tensor_info.hpp"
#include "bits512.hpp"
#include "common.hpp"
#include "errors.hpp"
#include <boost/align/is_aligned.hpp>

TCM_NAMESPACE_BEGIN

template <>
TCM_EXPORT auto obtain_tensor_info<bits512>(torch::Tensor x, char const* name)
    -> TensorInfo<bits512>
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
    TCM_CHECK(strides[0] % 8L == 0, std::invalid_argument,
              fmt::format("{}'s stride along the first dimension must be a "
                          "multiple of 8; got {}",
                          arg_name, strides[0]));

    auto* data = x.data_ptr<int64_t>();
    TCM_CHECK(boost::alignment::is_aligned(64U, data), std::invalid_argument,
              fmt::format("{} must be aligned to 64-byte boundary", arg_name));
    return {reinterpret_cast<bits512*>(data), sizes[0], strides[0] / 8L};
}

namespace detail {
template <class T, class = std::enable_if_t<is_complex_v<T>>>
auto complex_tensor_info_impl(torch::Tensor x, char const* name)
    -> TensorInfo<T>
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
    TCM_CHECK(strides[0] % 2L == 0, std::invalid_argument,
              fmt::format("{}'s stride along the first dimension must be a "
                          "multiple of 2; got {}",
                          arg_name, strides[0]));
    auto* data = x.data_ptr<typename T::value_type>();
    return {reinterpret_cast<T*>(data), sizes[0], strides[0] / 2L};
}
} // namespace detail

#define COMPLEX_IMPLEMENTATION(type)                                           \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type>(torch::Tensor x,                  \
                                             char const*   name)               \
        ->TensorInfo<type>                                                     \
    {                                                                          \
        return detail::complex_tensor_info_impl<type>(std::move(x), name);     \
    }
COMPLEX_IMPLEMENTATION(std::complex<float>)
COMPLEX_IMPLEMENTATION(std::complex<double>)
#undef COMPLEX_IMPLEMENTATION

template <class T, class = void> struct reinterpret_as {
    using type = T;
};
template <class T>
struct reinterpret_as<T, std::enable_if_t<std::is_unsigned_v<T>>> {
    using type = std::make_signed_t<T>;
};
template <class T> using reinterpret_as_t = typename reinterpret_as<T>::type;

#define TRIVIAL_IMPLEMENTATION(type)                                           \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type>(torch::Tensor x,                  \
                                             char const*   name)               \
        ->TensorInfo<type>                                                     \
    {                                                                          \
        auto const arg_name = name != nullptr ? name : "tensor";               \
        auto const sizes    = x.sizes();                                       \
        TCM_CHECK(                                                             \
            sizes.size() == 1, std::invalid_argument,                          \
            fmt::format(                                                       \
                "{} has wrong shape: [{}]; expected a one-dimensional tensor", \
                arg_name, fmt::join(sizes, ",")));                             \
        type* data;                                                            \
        using RawT = reinterpret_as_t<type>;                                   \
        if constexpr (!std::is_same_v<type, RawT>) {                           \
            data = reinterpret_cast<type*>(x.data_ptr<RawT>());                \
        }                                                                      \
        else {                                                                 \
            data = x.data_ptr<type>();                                         \
        }                                                                      \
        return {data, sizes[0], x.strides()[0]};                               \
    }
TRIVIAL_IMPLEMENTATION(float)
TRIVIAL_IMPLEMENTATION(double)
TRIVIAL_IMPLEMENTATION(int64_t)
TRIVIAL_IMPLEMENTATION(uint64_t)
#undef TRIVIAL_IMPLEMENTATION

#define ADD_CONST_OVERLOADS(type)                                              \
    template <>                                                                \
    TCM_EXPORT auto obtain_tensor_info<type const>(torch::Tensor x,            \
                                                   char const*   name)         \
        ->TensorInfo<type const>                                               \
    {                                                                          \
        return obtain_tensor_info<type>(std::move(x), name);                   \
    }
ADD_CONST_OVERLOADS(float)
ADD_CONST_OVERLOADS(double)
ADD_CONST_OVERLOADS(int64_t)
ADD_CONST_OVERLOADS(uint64_t)
ADD_CONST_OVERLOADS(bits512)
ADD_CONST_OVERLOADS(std::complex<float>)
ADD_CONST_OVERLOADS(std::complex<double>)
#undef ADD_CONST_OVERLOADS

TCM_NAMESPACE_END
