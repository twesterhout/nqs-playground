#include "bits512.hpp"

TCM_EXPORT auto operator==(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    for (auto i = 0; i < static_cast<int>(std::size(x.words)); ++i) {
        if (x.words[i] != y.words[i]) { return false; }
    }
    return true;
}

TCM_EXPORT auto operator!=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    return !(x == y);
}

TCM_EXPORT auto operator<(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    for (auto i = 0; i < static_cast<int>(std::size(x.words)); ++i) {
        if (x.words[i] < y.words[i]) { return true; }
        if (x.words[i] > y.words[i]) { return false; }
    }
    return false;
}

TCM_EXPORT auto operator>(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    return y < x;
}

TCM_EXPORT auto operator<=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    return !(x > y);
}

TCM_EXPORT auto operator>=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    return !(x < y);
}

TCM_NAMESPACE_BEGIN

namespace detail {
// Compiler-friendly rotate left function. Both GCC and Clang are clever enough
// to replace it with a `rol` instruction.
constexpr auto rotl64(uint64_t n, uint32_t c) noexcept -> uint64_t
{
    constexpr uint32_t mask = 8 * sizeof(n) - 1;
    c &= mask;
    return (n << c) | (n >> ((-c) & mask));
}

constexpr auto fmix64(uint64_t k) noexcept -> uint64_t
{
    k ^= k >> 33U;
    k *= 0xff51afd7ed558ccdLLU;
    k ^= k >> 33U;
    k *= 0xc4ceb9fe1a85ec53LLU;
    k ^= k >> 33U;
    return k;
}

constexpr auto murmurhash3_x64_128(uint64_t const (&words)[8], uint64_t (&out)[2]) noexcept -> void
{
    constexpr uint64_t c1   = 0x87c37b91114253d5LLU;
    constexpr uint64_t c2   = 0x4cf5ad432745937fLLU;
    constexpr uint32_t seed = 0x208546c8U;
    constexpr int      size = 64;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    for (auto i = 0; i < size / 16; ++i) {
        auto k1 = words[i * 2 + 0];
        auto k2 = words[i * 2 + 1];

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;

        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;

        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    // These are useless
    // h1 ^= size;
    // h2 ^= size;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    out[0] = h1;
    out[1] = h2;
}
} // namespace detail

TCM_NAMESPACE_END

namespace std {

TCM_EXPORT auto
hash<::TCM_NAMESPACE::bits512>::operator()(::TCM_NAMESPACE::bits512 const& x) const noexcept
    -> size_t
{
    uint64_t out[2];
    ::TCM_NAMESPACE::detail::murmurhash3_x64_128(x.words, out);
    // This part is questionable: should we mix the words in some way or is
    // this good enough...
    return out[0];
}

} // namespace std
