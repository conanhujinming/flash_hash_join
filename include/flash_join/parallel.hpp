#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

namespace flash_join::parallel {

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

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (std::size_t thread = 1; thread < threads; ++thread) {
        const auto begin = work_items * thread / threads;
        const auto end = work_items * (thread + 1) / threads;
        workers.emplace_back([&, thread, begin, end] { function(thread, begin, end); });
    }
    function(0, 0, work_items / threads);
    for (auto& worker : workers) {
        worker.join();
    }
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
