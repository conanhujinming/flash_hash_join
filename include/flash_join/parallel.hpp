#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace flash_join::parallel {

namespace detail {

class WorkerPool {
public:
    WorkerPool() = default;
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    ~WorkerPool() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    void ensure_workers(std::size_t count) {
        const std::lock_guard<std::mutex> lock(mutex_);
        while (workers_.size() < count) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void submit(std::function<void()> task) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        ready_.notify_one();
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

inline WorkerPool& worker_pool() {
    static WorkerPool pool;
    return pool;
}

class BatchState {
public:
    void add_task() noexcept {
        remaining_.fetch_add(1, std::memory_order_relaxed);
    }

    void finish_task() noexcept {
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            completed_.notify_one();
        }
    }

    void capture_current_exception() noexcept {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!exception_) {
            exception_ = std::current_exception();
        }
    }

    void wait_and_rethrow() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            completed_.wait(lock, [this] {
                return remaining_.load(std::memory_order_acquire) == 0;
            });
        }
        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

private:
    std::atomic<std::size_t> remaining_{0};
    std::mutex mutex_;
    std::condition_variable completed_;
    std::exception_ptr exception_;
};

}  // namespace detail

inline std::size_t effective_thread_count(
    std::size_t requested,
    std::size_t work_items,
    std::size_t minimum_grain = 1) noexcept {
    requested = std::max<std::size_t>(1, requested);
    if (work_items == 0) {
        return 1;
    }
    const auto useful = std::max<std::size_t>(1, (work_items + minimum_grain - 1) / minimum_grain);
    return std::min({requested, useful, work_items});
}

template <typename Function>
void for_chunks(std::size_t work_items, std::size_t threads, Function&& function) {
    threads = effective_thread_count(threads, work_items);
    if (threads == 1) {
        function(0, 0, work_items);
        return;
    }

    auto& pool = detail::worker_pool();
    pool.ensure_workers(threads - 1);
    detail::BatchState batch;
    for (std::size_t thread = 1; thread < threads; ++thread) {
        const auto begin = work_items * thread / threads;
        const auto end = work_items * (thread + 1) / threads;
        batch.add_task();
        try {
            pool.submit([&, thread, begin, end] {
                try {
                    function(thread, begin, end);
                } catch (...) {
                    batch.capture_current_exception();
                }
                batch.finish_task();
            });
        } catch (...) {
            batch.finish_task();
            batch.capture_current_exception();
            break;
        }
    }
    try {
        function(0, 0, work_items / threads);
    } catch (...) {
        batch.capture_current_exception();
    }
    batch.wait_and_rethrow();
}

template <typename Function>
void for_tasks(std::size_t task_count, std::size_t threads, Function&& function) {
    threads = effective_thread_count(threads, task_count);
    if (threads == 1) {
        for (std::size_t task = 0; task < task_count; ++task) {
            function(task);
        }
        return;
    }

    std::atomic<std::size_t> next_task{0};
    for_chunks(threads, threads, [&](std::size_t, std::size_t, std::size_t) {
        while (true) {
            const auto task = next_task.fetch_add(1, std::memory_order_relaxed);
            if (task >= task_count) {
                break;
            }
            function(task);
        }
    });
}

}  // namespace flash_join::parallel
