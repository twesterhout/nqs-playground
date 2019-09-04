// Copyright (c) 2019, Tom Westerhout
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

#include "spin.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <mutex>

TCM_NAMESPACE_BEGIN

namespace detail {
auto spin_configuration_to_string(gsl::span<float const> spin) -> std::string
{
    std::ostringstream msg;
    msg << '[';
    if (spin.size() > 0) {
        msg << spin[0];
        for (auto i = size_t{1}; i < spin.size(); ++i) {
            msg << ", " << spin[i];
        }
    }
    msg << ']';
    return msg.str();
}

namespace {
    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    auto copy_to_numpy_array(SpinVector const&                     spin,
                             pybind11::array_t<float, ExtraFlags>& out) -> void
    {
        TCM_CHECK_DIM(out.ndim(), 1);
        TCM_CHECK_SHAPE(out.shape(0), spin.size());

        auto const spin2float = [](Spin const s) noexcept->float
        {
            return s == Spin::up ? 1.0f : -1.0f;
        };

        auto access = out.template mutable_unchecked<1>();
        for (auto i = 0u; i < spin.size(); ++i) {
            access(i) = spin2float(spin[i]);
        }
    }
} // unnamed namespace
} // namespace detail

auto SpinVector::numpy() const
    -> pybind11::array_t<float, pybind11::array::c_style>
{
    pybind11::array_t<float, pybind11::array::c_style> out{size()};
    detail::copy_to_numpy_array(*this, out);
    return out;
}

auto SpinVector::tensor() const -> torch::Tensor
{
    auto out = detail::make_tensor<float>(size());
    copy_to({out.data<float>(), size()});
    return out;
}

SpinVector::SpinVector(gsl::span<float const> spin)
{
    check_range(spin);
    copy_from(spin);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

SpinVector::SpinVector(torch::Tensor const& spins)
{
    TCM_CHECK_DIM(spins.dim(), 1);
    TCM_CHECK(spins.is_contiguous(), std::invalid_argument,
              "input tensor must be contiguous");
    TCM_CHECK_TYPE(spins.scalar_type(), torch::kFloat32);
    auto buffer = gsl::span<float const>{spins.data<float>(),
                                         static_cast<size_t>(spins.size(0))};
    check_range(buffer);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

SpinVector::SpinVector(pybind11::str str)
{
    // PyUnicode_Check macro uses old style casts
#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
    // Borrowed from pybind11/pytypes.h
    pybind11::object temp = str;
    if (PyUnicode_Check(str.ptr())) {
        temp = pybind11::reinterpret_steal<pybind11::object>(
            PyUnicode_AsUTF8String(str.ptr()));
        if (!temp)
            throw std::runtime_error{
                "Unable to extract string contents! (encoding issue)"};
    }
    char*             buffer;
    pybind11::ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length)) {
        throw std::runtime_error{
            "Unable to extract string contents! (invalid type)"};
    }
    TCM_CHECK(length <= static_cast<pybind11::ssize_t>(max_size()),
              std::overflow_error,
              fmt::format("spin chain too long {}; expected <={}", length,
                          max_size()));

    _data.as_ints = _mm_set1_epi32(0);
    _data.size    = static_cast<std::uint16_t>(length);
    for (auto i = 0u; i < _data.size; ++i) {
        auto const s = buffer[i];
        TCM_CHECK(s == '0' || s == '1', std::domain_error,
                  fmt::format("invalid spin '{}'; expected '0' or '1'", s));
        unsafe_at(i) = (s == '1') ? Spin::up : Spin::down;
    }
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#endif
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

#if 0
auto unpack_to_tensor(gsl::span<SpinVector const> src, torch::Tensor dst)
    -> void
{
    // unpack_to_tensor(std::begin(src), std::end(src), dst,
    //                  [](auto const& x) -> SpinVector const& { return x; });
    if (src.empty()) { return; }

    auto const size         = src.size();
    auto const number_spins = src[0].size();
    TCM_ASSERT(std::all_of(std::begin(src), std::end(src),
                           [number_spins](auto const& x) {
                               return x.size() == number_spins;
                           }),
               "Input range contains variable size spin chains");
    TCM_ASSERT(dst.dim() == 2, fmt::format("Invalid dimension {}", dst.dim()));
    TCM_ASSERT(size == static_cast<size_t>(dst.size(0)),
               fmt::format("Sizes don't match: size={}, dst.size(0)={}", size,
                           dst.size(0)));
    TCM_ASSERT(static_cast<int64_t>(number_spins) == dst.size(1),
               fmt::format("Sizes don't match: number_spins={}, dst.size(1)={}",
                           number_spins, dst.size(1)));
    TCM_ASSERT(dst.is_contiguous(), "Output tensor must be contiguous");

    auto const chunks_16     = number_spins / 16;
    auto const rest_16       = number_spins % 16;
    auto const rest_8        = number_spins % 8;
    auto const copy_cheating = [chunks = chunks_16 + (rest_16 != 0)](
                                   SpinVector const& spin, float* out) {
        for (auto i = 0u; i < chunks; ++i, out += 16) {
            detail::unpack(spin._data.spin[i], out);
        }
    };

    auto* data = dst.data<float>();
    for (auto i = size_t{0}; i < size - 1; ++i, data += number_spins) {
        copy_cheating(src[i], data);
    }
    src[size - 1].copy_to({data, number_spins});
}
#endif

/// This is adapted from Boost.Math. All credit goes to Boost.Math developers.
constexpr auto unchecked_factorial(unsigned const x) noexcept -> long double
{
    constexpr std::array<long double, 171> factorials = {{
        1L,
        1L,
        2L,
        6L,
        24L,
        120L,
        720L,
        5040L,
        40320L,
        362880.0L,
        3628800.0L,
        39916800.0L,
        479001600.0L,
        6227020800.0L,
        87178291200.0L,
        1307674368000.0L,
        20922789888000.0L,
        355687428096000.0L,
        6402373705728000.0L,
        121645100408832000.0L,
        0.243290200817664e19L,
        0.5109094217170944e20L,
        0.112400072777760768e22L,
        0.2585201673888497664e23L,
        0.62044840173323943936e24L,
        0.15511210043330985984e26L,
        0.403291461126605635584e27L,
        0.10888869450418352160768e29L,
        0.304888344611713860501504e30L,
        0.8841761993739701954543616e31L,
        0.26525285981219105863630848e33L,
        0.822283865417792281772556288e34L,
        0.26313083693369353016721801216e36L,
        0.868331761881188649551819440128e37L,
        0.29523279903960414084761860964352e39L,
        0.103331479663861449296666513375232e41L,
        0.3719933267899012174679994481508352e42L,
        0.137637530912263450463159795815809024e44L,
        0.5230226174666011117600072241000742912e45L,
        0.203978820811974433586402817399028973568e47L,
        0.815915283247897734345611269596115894272e48L,
        0.3345252661316380710817006205344075166515e50L,
        0.1405006117752879898543142606244511569936e52L,
        0.6041526306337383563735513206851399750726e53L,
        0.265827157478844876804362581101461589032e55L,
        0.1196222208654801945619631614956577150644e57L,
        0.5502622159812088949850305428800254892962e58L,
        0.2586232415111681806429643551536119799692e60L,
        0.1241391559253607267086228904737337503852e62L,
        0.6082818640342675608722521633212953768876e63L,
        0.3041409320171337804361260816606476884438e65L,
        0.1551118753287382280224243016469303211063e67L,
        0.8065817517094387857166063685640376697529e68L,
        0.427488328406002556429801375338939964969e70L,
        0.2308436973392413804720927426830275810833e72L,
        0.1269640335365827592596510084756651695958e74L,
        0.7109985878048634518540456474637249497365e75L,
        0.4052691950487721675568060190543232213498e77L,
        0.2350561331282878571829474910515074683829e79L,
        0.1386831185456898357379390197203894063459e81L,
        0.8320987112741390144276341183223364380754e82L,
        0.507580213877224798800856812176625227226e84L,
        0.3146997326038793752565312235495076408801e86L,
        0.1982608315404440064116146708361898137545e88L,
        0.1268869321858841641034333893351614808029e90L,
        0.8247650592082470666723170306785496252186e91L,
        0.5443449390774430640037292402478427526443e93L,
        0.3647111091818868528824985909660546442717e95L,
        0.2480035542436830599600990418569171581047e97L,
        0.1711224524281413113724683388812728390923e99L,
        0.1197857166996989179607278372168909873646e101L,
        0.8504785885678623175211676442399260102886e102L,
        0.6123445837688608686152407038527467274078e104L,
        0.4470115461512684340891257138125051110077e106L,
        0.3307885441519386412259530282212537821457e108L,
        0.2480914081139539809194647711659403366093e110L,
        0.188549470166605025498793226086114655823e112L,
        0.1451830920282858696340707840863082849837e114L,
        0.1132428117820629783145752115873204622873e116L,
        0.8946182130782975286851441715398316520698e117L,
        0.7156945704626380229481153372318653216558e119L,
        0.5797126020747367985879734231578109105412e121L,
        0.4753643337012841748421382069894049466438e123L,
        0.3945523969720658651189747118012061057144e125L,
        0.3314240134565353266999387579130131288001e127L,
        0.2817104114380550276949479442260611594801e129L,
        0.2422709538367273238176552320344125971528e131L,
        0.210775729837952771721360051869938959523e133L,
        0.1854826422573984391147968456455462843802e135L,
        0.1650795516090846108121691926245361930984e137L,
        0.1485715964481761497309522733620825737886e139L,
        0.1352001527678402962551665687594951421476e141L,
        0.1243841405464130725547532432587355307758e143L,
        0.1156772507081641574759205162306240436215e145L,
        0.1087366156656743080273652852567866010042e147L,
        0.103299784882390592625997020993947270954e149L,
        0.9916779348709496892095714015418938011582e150L,
        0.9619275968248211985332842594956369871234e152L,
        0.942689044888324774562618574305724247381e154L,
        0.9332621544394415268169923885626670049072e156L,
        0.9332621544394415268169923885626670049072e158L,
        0.9425947759838359420851623124482936749562e160L,
        0.9614466715035126609268655586972595484554e162L,
        0.990290071648618040754671525458177334909e164L,
        0.1029901674514562762384858386476504428305e167L,
        0.1081396758240290900504101305800329649721e169L,
        0.1146280563734708354534347384148349428704e171L,
        0.1226520203196137939351751701038733888713e173L,
        0.132464181945182897449989183712183259981e175L,
        0.1443859583202493582204882102462797533793e177L,
        0.1588245541522742940425370312709077287172e179L,
        0.1762952551090244663872161047107075788761e181L,
        0.1974506857221074023536820372759924883413e183L,
        0.2231192748659813646596607021218715118256e185L,
        0.2543559733472187557120132004189335234812e187L,
        0.2925093693493015690688151804817735520034e189L,
        0.339310868445189820119825609358857320324e191L,
        0.396993716080872089540195962949863064779e193L,
        0.4684525849754290656574312362808384164393e195L,
        0.5574585761207605881323431711741977155627e197L,
        0.6689502913449127057588118054090372586753e199L,
        0.8094298525273443739681622845449350829971e201L,
        0.9875044200833601362411579871448208012564e203L,
        0.1214630436702532967576624324188129585545e206L,
        0.1506141741511140879795014161993280686076e208L,
        0.1882677176888926099743767702491600857595e210L,
        0.237217324288004688567714730513941708057e212L,
        0.3012660018457659544809977077527059692324e214L,
        0.3856204823625804217356770659234636406175e216L,
        0.4974504222477287440390234150412680963966e218L,
        0.6466855489220473672507304395536485253155e220L,
        0.8471580690878820510984568758152795681634e222L,
        0.1118248651196004307449963076076169029976e225L,
        0.1487270706090685728908450891181304809868e227L,
        0.1992942746161518876737324194182948445223e229L,
        0.269047270731805048359538766214698040105e231L,
        0.3659042881952548657689727220519893345429e233L,
        0.5012888748274991661034926292112253883237e235L,
        0.6917786472619488492228198283114910358867e237L,
        0.9615723196941089004197195613529725398826e239L,
        0.1346201247571752460587607385894161555836e242L,
        0.1898143759076170969428526414110767793728e244L,
        0.2695364137888162776588507508037290267094e246L,
        0.3854370717180072770521565736493325081944e248L,
        0.5550293832739304789551054660550388118e250L,
        0.80479260574719919448490292577980627711e252L,
        0.1174997204390910823947958271638517164581e255L,
        0.1727245890454638911203498659308620231933e257L,
        0.2556323917872865588581178015776757943262e259L,
        0.380892263763056972698595524350736933546e261L,
        0.571338395644585459047893286526105400319e263L,
        0.8627209774233240431623188626544191544816e265L,
        0.1311335885683452545606724671234717114812e268L,
        0.2006343905095682394778288746989117185662e270L,
        0.308976961384735088795856467036324046592e272L,
        0.4789142901463393876335775239063022722176e274L,
        0.7471062926282894447083809372938315446595e276L,
        0.1172956879426414428192158071551315525115e279L,
        0.1853271869493734796543609753051078529682e281L,
        0.2946702272495038326504339507351214862195e283L,
        0.4714723635992061322406943211761943779512e285L,
        0.7590705053947218729075178570936729485014e287L,
        0.1229694218739449434110178928491750176572e290L,
        0.2004401576545302577599591653441552787813e292L,
        0.3287218585534296227263330311644146572013e294L,
        0.5423910666131588774984495014212841843822e296L,
        0.9003691705778437366474261723593317460744e298L,
        0.1503616514864999040201201707840084015944e301L,
        0.2526075744973198387538018869171341146786e303L,
        0.4269068009004705274939251888899566538069e305L,
        0.7257415615307998967396728211129263114717e307L,
    }};

    return factorials[x];
}

constexpr auto max_factorial() noexcept -> unsigned { return 170; }

auto binomial_coefficient(unsigned const n, unsigned const k) -> uint64_t
{
    TCM_CHECK(k <= n, std::domain_error,
              fmt::format(
                  "binomial coefficient is undefined for k > n; got k={}, n={}",
                  k, n));
    if (k == 0 || k == n) { return 1; }
    if (k == 1 || k == n - 1) { return n; }
    TCM_CHECK(n <= max_factorial(), std::overflow_error,
              fmt::format("binomial coefficient is not (yet) implemented for "
                          "such big n values: n={}",
                          n));
    // Using fast table lookup:
    auto const result = unchecked_factorial(n) / unchecked_factorial(n - k)
                        / unchecked_factorial(k);
    TCM_CHECK(
        result < static_cast<long double>(std::numeric_limits<uint64_t>::max()),
        std::overflow_error,
        fmt::format(
            "binomial coefficient {} does not fit into a 64-bit integer",
            result));
    // convert to nearest integer:
    return static_cast<uint64_t>(std::ceil(result - 0.5L));
}

auto all_spins(unsigned n, optional<int> magnetisation)
    -> std::vector<SpinVector,
                   boost::alignment::aligned_allocator<SpinVector, 64>>
{
    using VectorT =
        std::vector<SpinVector,
                    boost::alignment::aligned_allocator<SpinVector, 64>>;
    if (magnetisation.has_value()) {
        TCM_CHECK(n <= SpinVector::max_size(), std::overflow_error,
                  fmt::format("invalid n: {}; expected <={}", n,
                              SpinVector::max_size()));
        TCM_CHECK(
            static_cast<unsigned>(std::abs(*magnetisation)) <= n,
            std::invalid_argument,
            fmt::format("magnetisation exceeds the number of spins: |{}| > {}",
                        *magnetisation, n));
        TCM_CHECK(
            (static_cast<int>(n) + *magnetisation) % 2 == 0, std::runtime_error,
            fmt::format("{} spins cannot have a magnetisation of {}. `n + "
                        "magnetisation` must be even",
                        n, *magnetisation));
        alignas(32) float buffer[SpinVector::max_size()];
        auto const        number_downs =
            static_cast<unsigned>((static_cast<int>(n) - *magnetisation) / 2);
        std::fill(buffer, buffer + number_downs, -1.0f);
        std::fill(buffer + number_downs, buffer + n, 1.0f);
        VectorT spins;
        spins.reserve(binomial_coefficient(n, number_downs));
        auto s = gsl::span<float>{&buffer[0], n};
        do {
            spins.emplace_back(s, UnsafeTag{});
        } while (std::next_permutation(std::begin(s), std::end(s)));
        return spins;
    }
    else {
        static_assert(SpinVector::max_size() >= 26,
                      TCM_STATIC_ASSERT_BUG_MESSAGE);
        TCM_CHECK(n <= 26, std::overflow_error,
                  fmt::format("too many spins: {}; refuse to allocate more "
                              "than 8GB of storage",
                              n));
        auto const size = 1UL << static_cast<size_t>(n);
        VectorT    spins;
        spins.reserve(size);
        for (auto i = size_t{0}; i < size; ++i) {
            spins.emplace_back(n, i);
        }
        return spins;
    }
}

auto SpinVector::numpy_dtype() -> pybind11::dtype
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        PYBIND11_NUMPY_DTYPE(SpinVector, _data.spin, _data.size);
    });
    return pybind11::dtype::of<SpinVector>();
}

auto bind_spin(pybind11::module m) -> void
{
    namespace py = pybind11;

    py::class_<SpinVector>(m, "CompactSpin", R"EOF(
        Compact representation of spin configurations. Each spin is encoded in a one bit.
    )EOF")
        .def(py::init<unsigned, uint64_t>(), py::arg{"size"}, py::arg{"data"},
             R"EOF(
                 Creates a compact spin configuration from bits packed into an integer.

                 :param size: number of spins. This parameter can't be deduced
                              from ``data``, because that would discard the leading
                              zeros.
                 :param data: a sequence of bits packed into an int. The value of the
                              ``i``'th spin is given by the ``i``'th most significant
                              bit of ``data``.
             )EOF")
        .def(py::init<torch::Tensor const&>(), py::arg{"x"},
             R"EOF(
                 Creates a compact spin configuration from a tensor.

                 :param x: a one-dimensional tensor of ``float``. ``-1.0`` means
                           spin down and ``1.0`` means spin up.
             )EOF")
        .def(py::init<py::str>(), py::arg{"x"},
             R"EOF(
                 Creates a compact spin configuration from a string.

                 :param x: a string consisting of '0's and '1's. '0' means spin
                           down and '1' means spin up.
             )EOF")
        .def(py::init<py::array_t<float, py::array::c_style> const&>(),
             py::arg{"x"},
             R"EOF(
                 Creates a compact spin configuration from a numpy array.

                 :param x: a one-dimensional contiguous array of ``float``. ``-1.0``
                           means spin down and ``1.0`` means spin up.
             )EOF")
        .def(
            "__copy__", [](SpinVector const& x) { return SpinVector{x}; },
            R"EOF(Copies the current spin configuration.)EOF")
        .def(
            "__deepcopy__",
            [](SpinVector const& x, py::dict /*unused*/) {
                return SpinVector{x};
            },
            py::arg{"memo"} = py::none(),
            R"EOF(Same as ``self.__copy__()``.)EOF")
        .def(
            "__len__", [](SpinVector const& self) { return self.size(); },
            R"EOF(
                 Returns the number of spins in the spin configuration.
             )EOF")
        .def(
            "__int__",
            [](SpinVector const& self) { return static_cast<size_t>(self); },
            R"EOF(
                 Implements ``int(self)``, i.e. conversion to ``int``.

                 .. warning::

                    This function does not work with spin configurations longer than 64.
             )EOF")
        .def(
            "__str__",
            [](SpinVector const& self) {
                return static_cast<std::string>(self);
            },
            py::return_value_policy::move,
            R"EOF(
                 Implements ``str(self)``, i.e. conversion to ``str``.
             )EOF")
        .def(
            "__getitem__",
            [](SpinVector const& x, unsigned const i) {
                return x.at(i) == Spin::up ? 1.0f : -1.0f;
            },
            py::arg{"i"},
            R"EOF(
                 Returns ``self[i]`` as a ``float``.
             )EOF")
        .def(
            "__setitem__",
            [](SpinVector& x, unsigned const i, float const spin) {
                auto const float2spin = [](auto const s) {
                    if (s == -1.0f) { return Spin::down; }
                    if (s == 1.0f) { return Spin::up; }
                    TCM_ERROR(
                        std::invalid_argument,
                        fmt::format("Invalid spin {}; expected either -1 or +1",
                                    s));
                };
                x.at(i) = float2spin(spin);
            },
            py::arg{"i"}, py::arg{"spin"},
            R"EOF(
                 Performs ``self[i] = spin``.

                 ``spin`` must be either ``-1.0`` or ``1.0``.
             )EOF")
        .def(
            "__eq__",
            [](SpinVector const& x, SpinVector const& y) { return x == y; },
            py::is_operator())
        .def(
            "__ne__",
            [](SpinVector const& x, SpinVector const& y) { return x != y; },
            py::is_operator())
        .def_property_readonly(
            "size", [](SpinVector const& self) { return self.size(); },
            R"EOF(Same as ``self.__len__()``.)EOF")
        .def_property_readonly(
            "magnetisation",
            [](SpinVector const& self) { return self.magnetisation(); },
            R"EOF(Returns the magnetisation)EOF")
        .def(
            "numpy", [](SpinVector const& x) { return x.numpy(); },
            py::return_value_policy::move,
            R"EOF(Converts the spin configuration to a numpy.ndarray)EOF")
        .def(
            "tensor", [](SpinVector const& x) { return x.tensor(); },
            py::return_value_policy::move,
            R"EOF(Converts the spin configuration to a torch.Tensor)EOF")
        .def("__hash__", &SpinVector::hash, R"EOF(
                Returns the hash of the spin configuration.
            )EOF")
        .def_property_readonly_static("dtype", [](py::object /*self*/) {
            return SpinVector::numpy_dtype();
        });

    m.def("random_spin",
          [](unsigned const size, optional<int> magnetisation) {
              auto& generator = global_random_generator();
              if (magnetisation.has_value()) {
                  return SpinVector::random(size, *magnetisation, generator);
              }
              else {
                  return SpinVector::random(size, generator);
              }
          },
          py::arg{"n"}, py::arg{"magnetisation"} = py::none(),
          R"EOF(
              Generates a random spin configuration.
          )EOF");

    m.def(
        "all_spins",
        [](unsigned n, optional<int> magnetisation) {
            auto       vector = all_spins(n, std::move(magnetisation));
            auto const data   = vector.data();
            auto const size   = vector.size();
            auto       base   = py::capsule{
                new decltype(vector){std::move(vector)},
                [](void* p) { delete static_cast<decltype(vector)*>(p); }};
            return py::array{SpinVector::numpy_dtype(), size, data, base};
        },
        py::arg{"n"}, py::arg{"magnetisation"});

    m.def(
        "unsafe_get",
        [](py::array_t<SpinVector, py::array::c_style> xs, size_t const i) {
            return *xs.data(i);
        },
        py::arg{"array"}.noconvert(), py::arg{"index"});

    m.def(
        "unpack",
        [](py::array_t<SpinVector, py::array::c_style> array) {
            TCM_CHECK(
                array.ndim() == 1, std::invalid_argument,
                fmt::format("`array` has incorrect dimension: {}; expected 1",
                            array.ndim()));
            auto data = array.data();
            auto size = array.shape(0);
            return unpack_to_tensor(data, data + size);
        },
        py::arg{"array"}.noconvert());

    m.def(
        "unpack",
        [](py::array_t<SpinVector, py::array::c_style> array,
           py::array_t<int64_t, py::array::c_style>    indices) {
            TCM_CHECK(
                array.ndim() == 1, std::invalid_argument,
                fmt::format("`array` has incorrect dimension: {}; expected 1",
                            array.ndim()));
            TCM_CHECK(
                indices.ndim() == 1, std::invalid_argument,
                fmt::format("`indices` has incorrect dimension: {}; expected 1",
                            indices.ndim()));
            auto first = indices.data();
            auto last  = first + indices.shape(0);
            TCM_CHECK(std::all_of(first, last,
                                  [n = array.shape(0)](auto const i) {
                                      return 0 <= i && i < n;
                                  }),
                      std::out_of_range,
                      "`indices` contains invalid incides for `array`");
            return unpack_to_tensor(
                first, last,
                [data = array.data()](auto const i) { return data[i]; });
        },
        py::arg{"array"}.noconvert(), py::arg{"indices"}.noconvert());
}

TCM_NAMESPACE_END
