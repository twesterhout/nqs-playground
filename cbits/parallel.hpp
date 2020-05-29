#include "config.hpp"
#include "errors.hpp"

// #include <ATen/Parallel.h>
#include <omp.h>

#if 1
#    include <condition_variable>
#    include <functional>
#    include <future>
#    include <memory>
#    include <mutex>
#    include <queue>
#    include <stdexcept>
#    include <thread>
#    include <vector>
#endif

TCM_NAMESPACE_BEGIN

#if 1
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
// for a single worker thread.
class ThreadPool {
  public:
    ThreadPool();

    template <class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>
    {
        using return_type = typename std::result_of<F()>::type;
        auto task         = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // don't allow enqueueing after stopping the pool
            TCM_CHECK(!stop, std::runtime_error,
                      "enqueue on stopped ThreadPool");
            tasks.emplace([p = std::move(task)]() { (*p)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool();

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

namespace detail {
class _ThreadPoolBase {
  public:
    _ThreadPoolBase();

    template <class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>
    {
        using return_type = typename std::result_of<F()>::type;
        auto task         = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            // don't allow enqueueing after stopping the pool
            TCM_CHECK(!_stop, std::runtime_error,
                      "enqueue on stopped ThreadPool");
            _tasks.emplace([p = std::move(task)]() { (*p)(); });
        }
        _condition.notify_one();
        return res;
    }

    auto run_one_task() -> bool;
    auto stop() -> void;

  private:
    std::queue<std::function<void()>> _tasks;
    std::mutex                        _queue_mutex;
    std::condition_variable           _condition;
    bool                              _stop;
};
} // namespace detail

template <class Function>
auto run_with_control_inversion(Function function) -> void
{
    detail::_ThreadPoolBase pool;
    std::exception_ptr      e_ptr = nullptr;

    auto body = [&pool, &e_ptr, function = std::move(function)]() {
        try {
            function([&pool](auto&& task) {
                return pool.enqueue(std::forward<decltype(task)>(task));
            });
        }
        catch (...) {
            e_ptr = std::current_exception();
        }
        pool.stop();
    };
    auto thread = std::thread{std::move(body)};

    // This thread is now the worker thread
    while (!pool.run_one_task()) {}
    thread.join();
    if (e_ptr) { std::rethrow_exception(e_ptr); }
}

namespace detail {
TCM_IMPORT auto global_thread_pool() noexcept -> ThreadPool&;
} // namespace detail
#endif

template <class F> auto async(F&& f)
{
#if 0
    using R     = typename std::result_of<F()>::type;
    auto task   = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto future = task->get_future();
    at::launch([p = std::move(task)]() { (*p)(); });
    return future;
#else
    return detail::global_thread_pool().enqueue(std::forward<F>(f));
#endif
}

struct omp_task_handler {
  private:
    std::atomic_flag   _error_flag    = ATOMIC_FLAG_INIT;
    std::exception_ptr _exception_ptr = nullptr;

  public:
    template <class F> auto submit(F f) -> void
    {
        static_assert(std::is_nothrow_copy_constructible<F>::value,
                      TCM_STATIC_ASSERT_BUG_MESSAGE);
#pragma omp task default(none) firstprivate(f)                                 \
    shared(_error_flag, _exception_ptr)
        {
            try {
                f();
            }
            catch (...) {
                if (!_error_flag.test_and_set()) {
                    _exception_ptr = std::current_exception();
                }
            }
        }
    }

    auto check_errors() const -> void
    {
        if (_exception_ptr) { std::rethrow_exception(_exception_ptr); }
    }
};

template <class Function, class Int>
auto omp_parallel_for(Function func, Int first, Int last, Int chunk_size)
    -> void
{
    std::atomic_flag   error_flag    = ATOMIC_FLAG_INIT;
    std::exception_ptr exception_ptr = nullptr;

#pragma omp parallel for schedule(dynamic, chunk_size) default(none)           \
    firstprivate(first, last, chunk_size)                                      \
        shared(func, error_flag, exception_ptr)
    for (auto i = first; i < last; ++i) {
        try {
            func(i);
        }
        catch (...) {
            if (!error_flag.test_and_set()) {
                exception_ptr = std::current_exception();
            }
        }
    }

    if (exception_ptr) { std::rethrow_exception(exception_ptr); }
}

TCM_NAMESPACE_END
