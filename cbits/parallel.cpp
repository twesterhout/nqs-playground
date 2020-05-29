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
TCM_EXPORT _ThreadPoolBase::_ThreadPoolBase()
    : _tasks{}, _queue_mutex{}, _condition{}, _stop{false}
{}

TCM_EXPORT auto _ThreadPoolBase::run_one_task() -> bool
{
    std::function<void()> task;

    {
        std::unique_lock<std::mutex> lock(_queue_mutex);
        _condition.wait(lock, [this] { return _stop || !_tasks.empty(); });
        if (_stop && _tasks.empty()) return true;
        task = std::move(_tasks.front());
        _tasks.pop();
    }

    task();
    return false;
}

TCM_EXPORT auto _ThreadPoolBase::stop() -> void
{
    {
        std::unique_lock<std::mutex> lock(_queue_mutex);
        _stop = true;
    }
    _condition.notify_all();
}
} // namespace detail

namespace detail {
TCM_EXPORT auto global_thread_pool() noexcept -> ThreadPool&
{
    static ThreadPool pool{};
    return pool;
}
} // namespace detail
#endif

TCM_NAMESPACE_END
