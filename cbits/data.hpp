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

#pragma once

#include "errors.hpp"
#include "spin_basis.hpp"
#include "random.hpp"

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

TCM_NAMESPACE_BEGIN

// ------------------------------- [DataSet] ------------------------------- {{{
class SpinDataset {
    torch::Tensor _spins;
    torch::Tensor _values;
    size_t        _number_samples;
    size_t        _number_spins;

  public:
    SpinDataset(torch::Tensor spins, torch::Tensor values,
                size_t const number_spins)
        : _spins{std::move(spins)}
        , _values{std::move(values)}
        , _number_samples{}
        , _number_spins{number_spins}
    {
        TCM_ASSERT(_spins.defined(), "spins must be a valid tensor");
        TCM_ASSERT(_values.defined(), "values must be a valid tensor");
        TCM_CHECK(_spins.scalar_type() == torch::kInt64, std::invalid_argument,
                  fmt::format("spins has wrong scalar type: {}; expected int64",
                              _spins.scalar_type()));
        TCM_CHECK(
            _spins.dim() == 1, std::invalid_argument,
            fmt::format("spins has wrong number of dimensions: {}; expected 1",
                        _spins.dim()));
        auto spins_shape  = _spins.sizes();
        auto values_shape = _values.sizes();
        TCM_CHECK(
            spins_shape[0] == values_shape[0], std::invalid_argument,
            fmt::format(
                "spins and values have incompatible shapes: [{}] and [{}]; "
                "sizes must match along the first dimensiton",
                fmt::join(spins_shape.begin(), spins_shape.end(), ", "),
                fmt::join(values_shape.begin(), values_shape.end(), ", ")));
        _number_samples = static_cast<size_t>(spins_shape[0]);
        TCM_CHECK(_spins.device().type() == torch::DeviceType::CPU,
                  std::invalid_argument,
                  fmt::format("spins tensor must reside on the CPU"));
        TCM_CHECK(_values.device().type() == torch::DeviceType::CPU,
                  std::invalid_argument,
                  fmt::format("values tensor must reside on the CPU"));
    }

    SpinDataset(SpinDataset const&)     = default;
    SpinDataset(SpinDataset&&) noexcept = default;
    auto operator=(SpinDataset const&) -> SpinDataset& = default;
    auto operator=(SpinDataset&&) noexcept -> SpinDataset& = default;

    constexpr auto number_spins() const noexcept { return _number_spins; }
    constexpr auto number_samples() const noexcept { return _number_samples; }

    auto fetch(int64_t const first, int64_t const last,
               torch::Device const device, bool const pin_memory) const
        -> std::tuple<torch::Tensor, torch::Tensor>
    {
        TCM_CHECK(0 <= first && first <= last
                      && last <= static_cast<int64_t>(number_samples()),
                  std::invalid_argument,
                  fmt::format("invalid range: [{}, {})", first, last));
        if (first == last) { return {{}, {}}; };
        auto ys = _values.slice(/*dim=*/0, /*start=*/first, /*end=*/last)
                      .to(_values.options().device(device),
                          /*non_blocking=*/true, /*copy=*/false);
        auto xs = unpack(_spins.slice(/*dim=*/0, /*start=*/first, /*end=*/last),
                         pin_memory);
        xs      = xs.to(xs.options().device(device),
                   /*non_blocking=*/true, /*copy=*/false);
        return {std::move(xs), std::move(ys)};
    }

    auto fetch(gsl::span<int64_t const> indices, torch::Device const device,
               bool const pin_memory) const
        -> std::tuple<torch::Tensor, torch::Tensor>
    {
        if (indices.empty()) { return {{}, {}}; };
        auto ys = _values.index({torch::from_blob(
            const_cast<void*>(static_cast<void const*>(indices.data())),
            {static_cast<int64_t>(indices.size())},
            torch::TensorOptions{}.dtype(torch::kInt64))});
        ys      = ys.to(ys.options().device(device),
                   /*non_blocking=*/true, /*copy=*/false);
        auto xs = unpack(_spins, indices, pin_memory);
        xs      = xs.to(xs.options().device(device),
                   /*non_blocking=*/true, /*copy=*/false);
        return {std::move(xs), std::move(ys)};
    }

  private:
    auto unpack(torch::Tensor const& spins, bool const pin_memory) const
        -> torch::Tensor
    {
        TCM_ASSERT(spins.dim() == 1, "spins has wrong dimension");
        return unpack(
            gsl::span<SpinBasis::StateT const>{
                static_cast<SpinBasis::StateT const*>(spins.data_ptr()),
                static_cast<size_t>(spins.size(0))},
            pin_memory);
    }

    auto unpack(gsl::span<SpinBasis::StateT const> spins,
                bool const pin_memory) const -> torch::Tensor
    {
        using std::begin, std::end;
        TCM_ASSERT(!spins.empty(), "spins must not be empty");
        auto out = _allocate_spins_buffer(spins.size(), pin_memory);
        v2::unpack(begin(spins), end(spins), number_spins(), out);
        return out;
    }

    auto unpack(torch::Tensor const& spins, gsl::span<int64_t const> indices,
                bool const pin_memory) const -> torch::Tensor
    {
        TCM_ASSERT(spins.dim() == 1, "spins has wrong dimension");
        return unpack(
            gsl::span<SpinBasis::StateT const>{
                static_cast<SpinBasis::StateT const*>(spins.data_ptr()),
                static_cast<size_t>(spins.size(0))},
            indices, pin_memory);
    }

    auto unpack(gsl::span<SpinBasis::StateT const> spins,
                gsl::span<int64_t const> indices, bool const pin_memory) const
        -> torch::Tensor
    {
        using std::begin, std::end;
        TCM_ASSERT(!indices.empty(), "spins must not be empty");
        auto       out = _allocate_spins_buffer(indices.size(), pin_memory);
        auto const projection = [p = spins.data(),
                                 n = static_cast<int64_t>(spins.size())](
                                    auto const index) {
            TCM_CHECK(0 <= index && index <= n, std::out_of_range,
                      fmt::format("indices contains an index which is out of "
                                  "range: {}; expected an index in [{}, {})",
                                  index, 0, n));
            return p[index];
        };
        v2::unpack(begin(indices), end(indices), number_spins(), out,
                   projection);
        return out;
    }

    auto _allocate_spins_buffer(size_t const size, bool const pin_memory) const
        -> torch::Tensor
    {
        return torch::empty(
            {static_cast<int64_t>(size), static_cast<int64_t>(number_spins())},
            torch::TensorOptions{}
                .dtype(torch::kFloat32)
                .pinned_memory(pin_memory));
    }
};

namespace detail {
// Copyright (c) 2012 Jakob Progsch, Václav Zeman
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
//    1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
//
//    2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
//
//    3. This notice may not be removed or altered from any source
//    distribution.
//
// The following is a small adaptation of https://github.com/progschj/ThreadPool
// got a single worker thread.
class ThreadPool {
  public:
    ThreadPool() : worker{}, tasks{}, queue_mutex{}, condition{}, stop{false}
    {
        worker = std::thread{[this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock,
                                   [this] { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }

                task();
            }
        }};
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task         = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // don't allow enqueueing after stopping the pool
            if (stop) throw std::runtime_error{"enqueue on stopped ThreadPool"};
            tasks.emplace([p = std::move(task)]() { (*p)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        worker.join();
    }

  private:
    // need to keep track of threads so we can join them
    std::thread worker;
    // task queue
    std::queue<std::function<void()>> tasks;
    // synchronization
    std::mutex              queue_mutex;
    std::condition_variable condition;
    bool                    stop;
};
} // namespace detail

class ChunkLoader {
    using BatchT = std::tuple<torch::Tensor, torch::Tensor>;

    SpinDataset                         _dataset;
    std::future<BatchT>                 _future;
    size_t                              _offset;
    size_t                              _chunk_size;
    torch::Device                       _device;
    bool                                _pin_memory;
    std::optional<std::vector<int64_t>> _indices;
    detail::ThreadPool                  _pool;

  public:
    ChunkLoader(SpinDataset&& dataset, size_t chunk_size, bool shuffle,
                torch::Device device, bool pin_memory)
        : _dataset{std::move(dataset)}
        , _future{}
        , _offset{0}
        , _chunk_size{chunk_size}
        , _device{device}
        , _pin_memory{pin_memory}
        , _indices{}
        , _pool{}
    {
        if (shuffle) {
            using std::begin, std::end;
            _indices.emplace();
            _indices->resize(dataset.number_samples());
            std::iota(begin(*_indices), end(*_indices), int64_t{0});
        }
    }

    auto reset() -> void
    {
        _offset = 0;
        if (_indices.has_value()) {
            using std::begin, std::end;
            std::shuffle(begin(*_indices), end(*_indices),
                         global_random_generator());
        }
        maybe_submit_task();
    }

    auto next() -> std::optional<BatchT>
    {
        TCM_ASSERT(_offset <= _dataset.number_samples(),
                   "pre-condition violated");
        if (!_future.valid()) { return std::nullopt; }
        auto batch = _future.get();
        maybe_submit_task();
        TCM_ASSERT(_offset <= _dataset.number_samples(),
                   "post-condition violated");
        return batch;
    }

    auto maybe_submit_task() -> void
    {
        TCM_ASSERT(!_future.valid(), "pre-condition violated");
        auto const size =
            std::min(_chunk_size, _dataset.number_samples() - _offset);
        if (size > 0) {
            if (_indices.has_value()) {
                _future = _pool.enqueue([this, offset = _offset, size]() {
                    auto indices = gsl::span<int64_t const>{
                        _indices->data() + static_cast<int64_t>(offset), size};
                    return _dataset.fetch(indices, _device, _pin_memory);
                });
            }
            else {
                _future = _pool.enqueue([this, offset = _offset, size]() {
                    return _dataset.fetch(static_cast<int64_t>(offset),
                                          static_cast<int64_t>(offset + size),
                                          _device, _pin_memory);
                });
            }
            _offset += size;
        }
    }
};

#if 0
/// \brief A poor man's alternative to `torch.ConcatDataset`.
///
/// Multiple Markov chains are simply concatenated.
class DataSet {
  private:
    // NOTE: We use shared_ptr's here because we share the ownership of the data
    // with Python code.
    std::vector<std::shared_ptr<ChainResult const>> _chunks;
    std::vector<size_t>                             _cum_sizes;

  public:
    /// Constructs a new dataset from multiple Markov chains.
    DataSet(std::vector<std::shared_ptr<ChainResult const>> chunks);

    DataSet(DataSet const&)     = default;
    DataSet(DataSet&&) noexcept = default;
    DataSet& operator=(DataSet const&) = default;
    DataSet& operator=(DataSet&&) noexcept = default;

    /// Returns the total number of samples in the dataset
    inline auto size() const noexcept -> size_t;

    /// Returns the number of spins in the system.
    ///
    /// It is assumed that all samples have the sample number of spins.
    inline auto number_spins() const noexcept -> size_t;

    /// Returns the `i`'th sample.
    ///
    /// \precondition `i < size()`.
    inline auto operator[](size_t i) const noexcept -> ChainState const&;

    /// Returns the `i`'th sample.
    ///
    /// \throws std::out_of_range if `i >= size()`.
    inline auto at(size_t i) const -> ChainState const&;

  private:
    template <class Iterator>
    auto check_valid(Iterator begin, Iterator end) -> void;
};

auto DataSet::size() const noexcept -> size_t { return _cum_sizes.back(); }

auto DataSet::number_spins() const noexcept -> size_t
{
    TCM_ASSERT(!_chunks.empty(), "number of chunks must be >0 by construction");
    return _chunks[0]->number_spins();
}

auto DataSet::operator[](size_t const i) const noexcept -> ChainState const&
{
    TCM_ASSERT(
        i < size(),
        noexcept_format("index out of bounds: {}; expected <={}", i, size()));
    if (_chunks.size() == 1) { return _chunks[0]->samples()[i]; }
    //
    // indices: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
    //           ^^^^^^^^^       ^            ^^^^^^^^
    // sizes:        5     ^^^^^ 1 ^^^^^^^^^^     3
    //                       3         4
    // cum_sizes:    5       8   9    13          16
    //                                  ^
    //                                  |
    //                                (3, 2)
    //
    // So if we want to get the 11'th sample, we first find the upper bound for
    // 11 in the _cum_sizes vector. This gives us chunk_index == 3. Then, to get
    // the inner index in the third chunk, we subtract _cum_sizes[chunk - 1]
    // (i.e. 9). Thus we arrive at: 2nd element in the 3rd chunk.
    auto const chunk_index = static_cast<size_t>(std::distance(
        std::begin(_cum_sizes),
        std::upper_bound(std::begin(_cum_sizes), std::end(_cum_sizes), i)));
    TCM_ASSERT(chunk_index < _chunks.size(), "");
    TCM_ASSERT(chunk_index == 0
                   || ((i >= _cum_sizes[chunk_index - 1])
                       && (i - _cum_sizes[chunk_index - 1]
                           < _chunks[chunk_index]->samples().size())),
               noexcept_format(
                   "i = {}, chunk_index = {}, _cum_sizes[chunk_index - 1] = "
                   "{}, _chunks[chunk_index]->samples().size() = {}",
                   i, chunk_index, _cum_sizes[chunk_index - 1],
                   _chunks[chunk_index]->samples().size()));
    return (chunk_index == 0)
               ? _chunks[0]->samples()[i]
               : _chunks[chunk_index]
                     ->samples()[i - _cum_sizes[chunk_index - 1]];
}

auto DataSet::at(size_t const i) const -> ChainState const&
{
    TCM_CHECK(i < size(), std::out_of_range,
              fmt::format("index out of range: {}; expected <{}", i, size()));
    return (*this)[i];
}
#endif
// ------------------------------- [DataSet] ------------------------------- }}}

// ---------------------------- [IndexSampler] ----------------------------- {{{
#if 0
/// Iterates over batches of indices.
class IndexSampler {
  private:
    /// All the indices of the data samples.
    std::vector<unsigned> _indices;
    /// Our current position in the `_indices` vector.
    size_t _index;
    /// Size of a batch of indices.
    size_t _batch_size;
    /// Whether to shuffle the indices.
    bool _shuffle;
    /// In case `_indices.size() % _batch_size != 0` the last chunk will be
    /// smaller than `_batch_size`. This parameter decides whether we should
    /// ignore the smaller chunk altogether.
    bool _ignore_last;

  public:
    /// Constructs a new sampler.
    ///
    /// \param size        Number of samples.
    /// \param batch_size  The desired batch size.
    /// \param shuffle     If true, indices will be shuffled on every #reset.
    ///                    Otherwise, indices will be processed in order.
    /// \param ignore_last If true, the last, smaller, batch will be ignored.
    IndexSampler(size_t size, size_t batch_size, bool shuffle,
                 bool ignore_last);

    IndexSampler(IndexSampler const&)     = default;
    IndexSampler(IndexSampler&&) noexcept = default;
    IndexSampler& operator=(IndexSampler const&) = default;
    IndexSampler& operator=(IndexSampler&&) noexcept = default;

    constexpr auto batch_size() const noexcept -> size_t { return _batch_size; }
    constexpr auto shuffle() const noexcept -> bool { return _shuffle; }
    constexpr auto ignore_last() const noexcept -> bool { return _ignore_last; }

    /// Resets the sampler.
    auto reset() -> void;

    /// Returns the next batch of indices.
    inline auto next() noexcept -> gsl::span<unsigned const>;
};

auto IndexSampler::next() noexcept -> gsl::span<unsigned const>
{
    TCM_ASSERT(_index <= _indices.size(),
               noexcept_format("{} > {}", _index, _indices.size()));
    auto const remaining_indices = _indices.size() - _index;
    if (remaining_indices == 0
        || (_ignore_last && remaining_indices < _batch_size)) {
        return {};
    }
    auto const size = std::min(remaining_indices, _batch_size);
    auto const result =
        gsl::span<unsigned const>{_indices.data() + _index, size};
    _index += size;
    TCM_ASSERT(
        _index <= _indices.size(),
        noexcept_format("{} > {}; size = ", _index, _indices.size(), size));
    return result;
}
#endif
// ---------------------------- [IndexSampler] ----------------------------- }}}

// ----------------------------- [DataLoader] ------------------------------ {{{
#if 0
class DataLoader {
  public:
    /// Types of transformations which can be applied to values.
    ///
    /// Using #Transform::Amplitude will result in applying `|.|` operation to
    /// all values. #Transform::Sign will result in applying the `signum`
    /// function to all the values. Moreover, the values will be returned as a
    /// tensor of `int64_t` rather than `float` so that they can be directly
    /// used to train a classifier.
    enum class Transform { No, Amplitude, Sign };

    /// An example on which to train.
    struct Example {
        /// Inputs
        torch::Tensor spins;
        /// Outputs
        torch::Tensor values;
        /// Number of times each value in `spins` was encountered during Monte
        /// Carlo sampling.
        ///
        /// \note It is currently not used anywhere but included here for
        /// completeness.
        torch::Tensor counts;

        Example()                   = default;
        Example(Example const&)     = default;
        Example(Example&&) noexcept = default;
        Example& operator=(Example const&) = default;
        Example& operator=(Example&&) noexcept = default;

        /// Allocates tensors for \p batch_size samples with system size
        /// `number_spins`. The type of the #values tensor is deduced from
        /// \p transform.
        inline Example(size_t batch_size, size_t number_spins,
                       DataLoader::Transform transform);

        inline Example(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>);
        Example(torch::Tensor s, torch::Tensor v, torch::Tensor c);

        /// Slices the sample along the zeroth dimension.
        inline auto slice(size_t first, size_t last) const -> Example;

        /// Compares two #Examples for equality. However, examples are not
        /// compared in the mathematical sense. Rather, it is checked whether
        /// examples "point" to the same position in the dataset. Obviously,
        /// this means that #kind_of_equal will only work correctly on examples
        /// from the same dataset, but it's only used by #Iterator, so that
        /// should be fine.
        friend inline auto kind_of_equal(Example const& x, Example const& y)
            -> bool;

      private:
        inline Example(UnsafeTag, torch::Tensor s, torch::Tensor v, torch::Tensor c);
    };

    struct Iterator;

    /// Marks the end of iteration.
    struct Sentinel {
        constexpr auto operator==(Sentinel const& other) noexcept -> bool;
        constexpr auto operator!=(Sentinel const& other) noexcept -> bool;
        constexpr auto operator==(Iterator const& other) noexcept -> bool;
        constexpr auto operator!=(Iterator const& other) noexcept -> bool;
    };

    struct Iterator {
        using value_type        = Example;
        using reference         = Example const&;
        using pointer           = Example const*;
        using iterator_category = std::input_iterator_tag;

        friend class DataLoader;

      private:
        DataLoader*    _loader;
        Example const* _batch;

      public:
        constexpr Iterator() noexcept : _loader{nullptr}, _batch{nullptr} {}
        constexpr Iterator(Iterator const&) noexcept = default;
        constexpr Iterator(Iterator&&) noexcept      = default;
        constexpr Iterator& operator=(Iterator const&) noexcept = default;
        constexpr Iterator& operator=(Iterator&&) noexcept = default;
        inline auto         operator++() -> Iterator&;
        inline auto         operator++(int);
        constexpr auto      operator*() const noexcept -> reference;
        constexpr auto      operator-> () const noexcept -> pointer;
        constexpr auto      operator==(Iterator const& other) const -> bool;
        constexpr auto      operator!=(Iterator const& other) const -> bool;
        constexpr auto operator==(Sentinel const& other) const noexcept -> bool;
        constexpr auto operator!=(Sentinel const& other) const noexcept -> bool;

      private:
        constexpr Iterator(DataLoader& loader, Example const* batch) noexcept;
    };

  private:
    DataSet                        _dataset;
    IndexSampler                   _sampler;
    Example                        _batch;
    std::vector<ChainState const*> _temp_buffer;
    Transform                      _transform;

  public:
    DataLoader(DataSet&& dataset, IndexSampler&& sampler, Transform transform);

    DataLoader(DataLoader const&) = delete;
    DataLoader(DataLoader&&)      = delete;
    DataLoader& operator=(DataLoader const&) = delete;
    DataLoader& operator=(DataLoader&&) = delete;

    auto reset() -> void;
    auto next() -> Example const*;

    inline auto begin() -> Iterator;
    inline auto end() -> Iterator;
};

auto DataLoader::begin() -> Iterator { return {*this, next()}; }

auto DataLoader::end() -> Iterator { return {*this, nullptr}; }

// ------------------------------ [Example] -------------------------------- {{{
DataLoader::Example::Example(size_t const batch_size, size_t const number_spins,
                             DataLoader::Transform const transform)
    : spins{detail::make_tensor<float>(batch_size, number_spins)}
    , values{transform == DataLoader::Transform::Sign
                 ? detail::make_tensor<int64_t>(batch_size)
                 : detail::make_tensor<float>(batch_size)}
    , counts{detail::make_tensor<int64_t>(batch_size)}
{}

DataLoader::Example::Example(UnsafeTag, torch::Tensor s, torch::Tensor v,
                             torch::Tensor c)
    : spins{std::move(s)}, values{std::move(v)}, counts{std::move(c)}
{
    TCM_ASSERT(spins.defined(), "");
    TCM_ASSERT(values.defined(), "");
    TCM_ASSERT(counts.defined(), "");
    TCM_ASSERT(spins.dim() == 2 && values.dim() == 1 && counts.dim() == 1, "");
    TCM_ASSERT(
        spins.size(0) == values.size(0) && spins.size(0) == counts.size(0), "");
    TCM_ASSERT(spins.scalar_type() == torch::kFloat32
               && counts.scalar_type() == torch::kInt64
               && (values.scalar_type() == torch::kFloat32
                   || values.scalar_type() == torch::kInt64), "");
}

DataLoader::Example::Example(
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensors)
    : Example{std::move(std::get<0>(tensors)), std::move(std::get<1>(tensors)),
              std::move(std::get<2>(tensors))}
{}

auto DataLoader::Example::slice(size_t const first, size_t const last) const
    -> Example
{
    TCM_ASSERT(spins.defined(), "");
    TCM_ASSERT(values.defined(), "");
    TCM_ASSERT(counts.defined(), "");
    auto const b = static_cast<int64_t>(first);
    auto const e = static_cast<int64_t>(last);
    return Example{UnsafeTag{}, spins.slice(/*dim=*/0, b, e),
                   values.slice(/*dim=*/0, b, e),
                   counts.slice(/*dim=*/0, b, e)};
}

auto kind_of_equal(DataLoader::Example const& x, DataLoader::Example const& y)
    -> bool
{
    if (x.spins.data_ptr() == y.spins.data_ptr()) {
        TCM_ASSERT(x.values.data_ptr() == y.values.data_ptr(), "");
        TCM_ASSERT(x.counts.data_ptr() == y.counts.data_ptr(), "");
        return true;
    }
    return false;
}
// ------------------------------ [Example] -------------------------------- }}}

// ------------------------------ [Sentinel] ------------------------------- {{{
constexpr auto DataLoader::Sentinel::operator==(Sentinel const& other) noexcept
    -> bool
{
    return true;
}

constexpr auto DataLoader::Sentinel::operator!=(Sentinel const& other) noexcept
    -> bool
{
    return false;
}

constexpr auto DataLoader::Sentinel::operator==(Iterator const& other) noexcept
    -> bool
{
    return other == *this;
}

constexpr auto DataLoader::Sentinel::operator!=(Iterator const& other) noexcept
    -> bool
{
    return other != *this;
}
// ------------------------------ [Sentinel] ------------------------------- }}}

// ------------------------------ [Iterator] ------------------------------- {{{
constexpr DataLoader::Iterator::Iterator(DataLoader&    loader,
                                         Example const* batch) noexcept
    : _loader{&loader}, _batch{batch}
{}

auto DataLoader::Iterator::operator++() -> Iterator&
{
    TCM_ASSERT(_loader != nullptr && _batch != nullptr,
               "iterator not incrementable");
    _batch = _loader->next();
    return *this;
}

auto DataLoader::Iterator::operator++(int)
{
    TCM_ASSERT(_loader != nullptr && _batch != nullptr,
               "iterator not incrementable");
    // NOTE(twesterhout): This is a hack.
    //
    // LegacyInputIterator concept requires that for an lvalue `x`
    // `(void)x++` should be equivalent to `(void)++x`. This can be done
    // by simply returning `void` in `operator++(int)`. However, there is
    // another requirement that says that `*x++` should be equivalent to
    // `value_type t = *x; ++x; return t`. So we define a wrapper type
    // with a single operation: `operator*` which holds `t`.
    struct Wrapper {
        value_type x;

        Wrapper(value_type const& value) : x{value} {}
        Wrapper(Wrapper const&) = delete;
        Wrapper(Wrapper&&) =
            default; // Before C++17 we need this to return by value
        Wrapper& operator=(Wrapper const&) = delete;
        Wrapper& operator=(Wrapper&&) = delete;

        auto     operator*()
            && noexcept(std::is_nothrow_move_constructible<value_type>::value)
                   -> value_type
        {
            return std::move(x);
        }
    };

    Wrapper wrapper{*_batch};
    ++(*this);
    return wrapper;
}

constexpr auto DataLoader::Iterator::operator*() const noexcept -> reference
{
    TCM_ASSERT(_batch != nullptr, "iterator not dereferenceable");
    return *_batch;
}

constexpr auto DataLoader::Iterator::operator-> () const noexcept -> pointer
{
    TCM_ASSERT(_batch != nullptr, "iterator not dereferenceable");
    return _batch;
}

constexpr auto DataLoader::Iterator::operator==(Iterator const& other) const
    -> bool
{
    TCM_ASSERT(_loader != nullptr && _loader == other._loader,
               "iterators pointing to different DataLoader's cannot be "
               "compared");
    // Very rarely do we compare two iterators where neither one of them is
    // the one-past-the-end iterator.
    if (TCM_UNLIKELY(_batch != nullptr && other._batch != nullptr)) {
        return kind_of_equal(*_batch, *other._batch);
    }
    return _batch == nullptr && other._batch == nullptr;
}

constexpr auto DataLoader::Iterator::operator!=(Iterator const& other) const
    -> bool
{
    return !(*this == other);
}

constexpr auto DataLoader::Iterator::operator==(Sentinel const& other) const
    noexcept -> bool
{
    TCM_ASSERT(_loader != nullptr, "iterators are not comparable");
    return _batch == nullptr;
}

constexpr auto DataLoader::Iterator::operator!=(Sentinel const& other) const
    noexcept -> bool
{
    return !(*this == other);
}
// ------------------------------ [Iterator] ------------------------------- }}}
#endif
// ----------------------------- [DataLoader] ------------------------------ }}}

#if 0
auto bind_dataloader(pybind11::module m) -> void;
#endif

TCM_NAMESPACE_END

#if 0
// We make Python think that our `tcm::DataLoader::Example`s are just tuples.
// It's nice, because one can then write loops like
// ```{.py}
// for x, y, _ in dataloader:
//     ... loss(module(x), y) ...
// ```
namespace pybind11 {
namespace detail {
    template <> struct type_caster<::TCM_NAMESPACE::DataLoader::Example> {
      public:
        PYBIND11_TYPE_CASTER(::TCM_NAMESPACE::DataLoader::Example,
                             _("Example"));

        auto load(handle src, bool convert) -> bool
        {
            using TupleT =
                std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
            type_caster<TupleT> caster;
            if (!caster.load(src, convert)) { return false; }
            value = ::TCM_NAMESPACE::DataLoader::Example{
                static_cast<TupleT&&>(std::move(caster))};
            return true;
        }

        static handle cast(::TCM_NAMESPACE::DataLoader::Example src,
                           return_value_policy /* policy */,
                           handle /* parent */)
        {
            return py::cast(std::make_tuple(src.spins, src.values, src.counts))
                .release();
        }
    };
} // namespace detail
} // namespace pybind11
#endif
