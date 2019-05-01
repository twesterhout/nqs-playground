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

#include "data.hpp"
#include "random.hpp"

TCM_NAMESPACE_BEGIN

// ------------------------------- [DataSet] ------------------------------- {{{
template <class Iterator>
auto DataSet::check_valid(Iterator begin, Iterator end) -> void
{
    TCM_CHECK(begin != end, std::invalid_argument,
              "list of chunks must not be empty");
    TCM_CHECK(std::all_of(
                  begin, end,
                  [](auto const& p) { return p != nullptr && p->size() != 0; }),
              std::invalid_argument,
              "all chunks must be neither nullptr or empty");
    TCM_CHECK(std::all_of(begin + 1, end,
                          [n = (*begin)->number_spins()](auto const& p) {
                              return p->number_spins() == n;
                          }),
              std::invalid_argument,
              "all chunks must correspond to the same system size");
}

DataSet::DataSet(std::vector<std::shared_ptr<ChainResult const>> chunks)
    : _chunks{std::move(chunks)}, _cum_sizes{}
{
    check_valid(std::begin(_chunks), std::end(_chunks));
    _cum_sizes.reserve(_chunks.size());
    auto sum = size_t{0};
    for (auto const& x : _chunks) {
        sum += x->size();
        _cum_sizes.push_back(sum);
    }
    TCM_ASSERT(_chunks.size() == _cum_sizes.size(), "");
}
// ------------------------------- [DataSet] ------------------------------- }}}

// ---------------------------- [IndexSampler] ----------------------------- {{{
IndexSampler::IndexSampler(size_t const size, size_t const batch_size,
                           bool const shuffle, bool const ignore_last)
    : _indices{}
    , _index{0}
    , _batch_size{batch_size}
    , _shuffle{shuffle}
    , _ignore_last{ignore_last}
{
    _indices.reserve(size);
    for (auto i = size_t{0}; i < size; ++i) {
        _indices.push_back(i);
    }
    if (_shuffle) {
        std::shuffle(std::begin(_indices), std::end(_indices),
                     global_random_generator());
    }
}

auto IndexSampler::reset() -> void
{
    _index = 0;
    if (_shuffle) {
        std::shuffle(begin(_indices), end(_indices), global_random_generator());
    }
}
// ---------------------------- [IndexSampler] ----------------------------- }}}

// ----------------------------- [DataLoader] ------------------------------ {{{
DataLoader::Example::Example(torch::Tensor s, torch::Tensor v, torch::Tensor c)
    : spins{s}, values{v}, counts{c}
{
    TCM_ASSERT(spins.defined(), "");
    TCM_ASSERT(values.defined(), "");
    TCM_ASSERT(counts.defined(), "");
    TCM_CHECK_DIM(spins.dim(), 2);
    TCM_CHECK_TYPE(spins.scalar_type(), torch::kFloat32);
    TCM_CHECK(spins.is_contiguous(), std::invalid_argument,
              "spins tensor must be contiguous");
    TCM_CHECK_DIM(values.dim(), 1);
    // TCM_CHECK_TYPE(values.scalar_type(), torch::kFloat32);
    TCM_CHECK(values.is_contiguous(), std::invalid_argument,
              "values tensor must be contiguous");
    TCM_CHECK_DIM(counts.dim(), 1);
    TCM_CHECK_TYPE(counts.scalar_type(), torch::kInt64);
    TCM_CHECK(counts.is_contiguous(), std::invalid_argument,
              "counts tensor must be contiguous");
    TCM_CHECK(spins.size(0) == values.size(0)
                  && spins.size(0) == counts.size(0),
              std::invalid_argument,
              fmt::format("leading dimensions do not match: {} vs. {} vs. {}",
                          spins.size(0), values.size(0), counts.size(0)));
}

DataLoader::DataLoader(DataSet&& dataset, IndexSampler&& sampler,
                       Transform const transform)
    : _dataset{std::move(dataset)}
    , _sampler{std::move(sampler)}
    , _temp_buffer{}
    , _transform{transform}
{
    _batch =
        Example{_sampler.batch_size(), _dataset.number_spins(), _transform};
    _temp_buffer.reserve(_sampler.batch_size());
}

auto DataLoader::reset() -> void
{
    _sampler.reset();
    _batch =
        Example{_sampler.batch_size(), _dataset.number_spins(), _transform};
}

auto DataLoader::next() -> Example const*
{
    auto indices = _sampler.next();
    if (indices.empty()) { return nullptr; }
    TCM_ASSERT(_batch.spins.defined(), "");
    TCM_ASSERT(_batch.values.defined(), "");
    TCM_ASSERT(_batch.counts.defined(), "");

    if (indices.size() != _sampler.batch_size()) {
        TCM_ASSERT(indices.size() < _sampler.batch_size(), "");
        _batch = _batch.slice(0, indices.size());
    }
    _temp_buffer.resize(indices.size());

    // TODO(twesterhout): This switch is ugly!
    switch (_transform) {
    case Transform::Amplitude: {
        auto values_accessor = _batch.values.template accessor<float, 1>();
        auto counts_accessor = _batch.counts.template accessor<int64_t, 1>();
        for (auto i = size_t{0}; i < indices.size(); ++i) {
            auto const& sample = _dataset[indices[i]];
            _temp_buffer[i]    = &sample;
            values_accessor[static_cast<int64_t>(i)] =
                static_cast<float>(std::abs(sample.value));
            counts_accessor[static_cast<int64_t>(i)] =
                static_cast<int64_t>(sample.count);
        }
        break;
    }
    case Transform::Sign: {
        auto values_accessor = _batch.values.template accessor<int64_t, 1>();
        auto counts_accessor = _batch.counts.template accessor<int64_t, 1>();
        for (auto i = size_t{0}; i < indices.size(); ++i) {
            auto const& sample = _dataset[indices[i]];
            _temp_buffer[i]    = &sample;
            values_accessor[static_cast<int64_t>(i)] =
                static_cast<int64_t>(sample.value < 0.0);
            counts_accessor[static_cast<int64_t>(i)] =
                static_cast<int64_t>(sample.count);
        }
        break;
    }
    case Transform::No: {
        auto values_accessor = _batch.values.template accessor<float, 1>();
        auto counts_accessor = _batch.counts.template accessor<int64_t, 1>();
        for (auto i = size_t{0}; i < indices.size(); ++i) {
            auto const& sample = _dataset[indices[i]];
            _temp_buffer[i]    = &sample;
            values_accessor[static_cast<int64_t>(i)] =
                static_cast<float>(sample.value);
            counts_accessor[static_cast<int64_t>(i)] =
                static_cast<int64_t>(sample.count);
        }
        break;
    }
    } // end switch

    // Loading spins
    unpack_to_tensor(
        std::begin(_temp_buffer), std::end(_temp_buffer), _batch.spins,
        [](auto const* x) -> SpinVector const& { return x->spin; });
    return &_batch;
}
// ----------------------------- [DataLoader] ------------------------------ }}}

auto bind_dataloader(pybind11::module m) -> void
{
    namespace py = pybind11;
    py::class_<DataSet>(m, "DataSet")
        .def(py::init<std::vector<std::shared_ptr<ChainResult const>>>())
        .def_property_readonly("size", &DataSet::size)
        .def_property_readonly("number_spins", &DataSet::number_spins)
        .def("__getitem__",
             [](DataSet const& self, size_t const i) { return self.at(i); });

    py::class_<DataLoader> dataloader{m, "DataLoader"};
    dataloader
        .def(py::init([](std::vector<std::shared_ptr<ChainResult const>> chunks,
                         DataLoader::Transform transform, size_t batch_size,
                         bool shuffle, bool ignore_last) {
                 DataSet      dataset{std::move(chunks)};
                 IndexSampler sampler{dataset.size(), batch_size, shuffle,
                                      ignore_last};
                 return std::make_unique<DataLoader>(
                     std::move(dataset), std::move(sampler), transform);
             }),
             py::arg{"dataset"}, py::arg{"transform"}, py::arg{"batch_size"},
             py::arg{"shuffle"} = true, py::arg{"ignore_last"} = false)
        .def("__iter__",
             [](DataLoader& self) {
                 self.reset();
                 return py::make_iterator(self.begin(), DataLoader::Sentinel{});
             },
             py::keep_alive<0, 1>());
    py::enum_<DataLoader::Transform>(dataloader, "Transform")
        .value("No", DataLoader::Transform::No)
        .value("Amplitude", DataLoader::Transform::Amplitude)
        .value("Sign", DataLoader::Transform::Sign)
        .export_values();
}

TCM_NAMESPACE_END
