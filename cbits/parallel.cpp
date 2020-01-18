#include "parallel.hpp"

TCM_NAMESPACE_BEGIN

auto global_executor() noexcept -> tf::Executor&
{
    static tf::Executor executor;
    return executor;
}

ThreadPool::ThreadPool()
    : worker{}, tasks{}, queue_mutex{}, condition{}, stop{false}
{
    worker = std::thread{[this] {
        for (;;) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this] { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
            }

            task();
        }
    }};
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    worker.join();
}

namespace detail {
auto global_thread_pool() noexcept -> ThreadPool&
{
    static ThreadPool pool{};
    return pool;
}
} // namespace detail

TCM_NAMESPACE_END
