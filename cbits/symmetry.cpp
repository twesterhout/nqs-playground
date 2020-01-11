#include "symmetry.hpp"

TCM_NAMESPACE_BEGIN

namespace detail {
static_assert(round_up_pow_2(1) == 1, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(2) == 2, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(3) == 4, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(4) == 4, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(5) == 8, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(123412398) == 134217728,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(32028753) == 33554432,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(0xFFFFFFF) == 0x10000000,
              TCM_STATIC_ASSERT_BUG_MESSAGE);

static_assert(log2(1) == 0, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(2) == 1, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(3) == 1, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(4) == 2, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(5) == 2, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(123498711) == 26, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(419224229) == 28, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(0xFFFFFFFF) == 31, TCM_STATIC_ASSERT_BUG_MESSAGE);

static_assert(bit_permute_step(5930UL, 272UL, 2U) == 5930UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(bit_permute_step(5930UL, 65UL, 1U) == 5929UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(bit_permute_step(56166UL, 2820UL, 4U) == 63846UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(bit_permute_step(13658UL, 242UL, 8U) == 22328UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);

constexpr auto test_benes_network() noexcept -> int
{
    constexpr BenesNetwork<uint8_t> p1{{0, 0, 1}, {5, 1, 0}};
    static_assert(p1(0) == 0, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(182) == 171, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(255) == 255, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(254) == 239, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(101) == 114, TCM_STATIC_ASSERT_BUG_MESSAGE);

    constexpr BenesNetwork<uint32_t> p2{
        {1162937344U, 304095283U, 67502857U, 786593U, 17233U},
        {16793941U, 18882595U, 263168U, 18U, 0U}};
    static_assert(p2(1706703868U) == 2923286909U,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p2(384262095U) == 1188297710U, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p2(991361073U) == 1634567835U, TCM_STATIC_ASSERT_BUG_MESSAGE);

    return 0;
}
} // namespace detail

namespace {
auto tensor_to_network(torch::Tensor x) -> detail::BenesNetwork<Symmetry::UInt>
{
    static_assert(std::is_same<Symmetry::UInt, uint64_t>::value,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    TCM_CHECK_TYPE("input tensor", x, torch::kInt64);
    TCM_CHECK_CONTIGUOUS("input tensor", x);

    using MasksT        = typename detail::BenesNetwork<Symmetry::UInt>::MasksT;
    constexpr auto size = 2 * std::tuple_size<MasksT>::value;
    TCM_CHECK_SHAPE("input tensor", x, {size});
    x = x.cpu();

    detail::BenesNetwork<Symmetry::UInt> out;
    auto const* first  = reinterpret_cast<uint64_t const*>(x.data_ptr());
    auto const* middle = first + out.fwd.size();
    auto const* last   = middle + out.bwd.size();
    std::copy(first, middle, std::begin(out.fwd));
    std::copy(middle, last, std::begin(out.bwd));
    return out;
}
} // namespace

TCM_EXPORT Symmetry::Symmetry(detail::BenesNetwork<UInt> const permute,
                              unsigned const sector, unsigned const periodicity)
    : _permute{permute}, _sector{sector}, _periodicity{periodicity}
{
    TCM_CHECK(
        periodicity > 0, std::invalid_argument,
        fmt::format("invalid periodicity: {}; expected a positive integer",
                    periodicity));
    TCM_CHECK(sector < periodicity, std::invalid_argument,
              fmt::format("invalid sector: {}; expected an integer in [0, {})",
                          sector, periodicity));
}

TCM_NAMESPACE_END
