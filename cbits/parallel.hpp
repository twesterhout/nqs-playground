#include "config.hpp"
#include <taskflow/taskflow.hpp>

TCM_NAMESPACE_BEGIN

inline auto global_executor() noexcept -> tf::Executor&
{
    static tf::Executor executor;
    return executor;
}

#if 0
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
#endif

TCM_NAMESPACE_END
