#include "parallel.hpp"

TCM_NAMESPACE_BEGIN

#if 1
TCM_EXPORT ThreadPool::ThreadPool()
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

TCM_EXPORT ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    worker.join();
}

namespace detail {
TCM_EXPORT auto global_thread_pool() noexcept -> ThreadPool&
{
    static ThreadPool pool{};
    return pool;
}
} // namespace detail
#endif

TCM_NAMESPACE_END
