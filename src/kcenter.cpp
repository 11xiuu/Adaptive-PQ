#include "kcenter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_progress(const char* name, std::int64_t current, std::int64_t total) {
    if (total <= 0) {
        return;
    }

    const int width = 40;
    const double ratio = static_cast<double>(current) / static_cast<double>(total);
    const int filled = static_cast<int>(ratio * width);

    std::cout << '\r' << name << " [";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '.');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << '%';

    if (current >= total) {
        std::cout << '\n';
    }
    std::cout.flush();
}

float dist_sq(
    const Dataset& data,
    std::int64_t a,
    std::int64_t b,
    const std::vector<std::int64_t>& dims
) {
    const float* x = data.data_ptr + a * data.d;
    const float* y = data.data_ptr + b * data.d;
    float sum = 0.0f;

    for (const std::int64_t dim : dims) {
        const float diff = x[dim] - y[dim];
        sum += diff * diff;
    }

    return sum;
}

std::vector<std::int64_t> run_block_kcenter(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    std::int64_t begin,
    std::int64_t end,
    float radius_sq
) {
    std::vector<std::int64_t> centers;

    if (begin >= end) {
        return centers;
    }

    centers.push_back(begin);

    for (std::int64_t i = begin + 1; i < end; ++i) {
        bool covered = false;

        for (const std::int64_t center : centers) {
            if (dist_sq(data, i, center, dims) <= radius_sq) {
                covered = true;
                break;
            }
        }

        if (!covered) {
            centers.push_back(i);
        }
    }

    return centers;
}

std::vector<std::int64_t> farthest_first_order(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    const std::vector<std::int64_t>& candidates
) {
    if (candidates.empty()) {
        return {};
    }

    const std::int64_t reorder_limit = std::min<std::int64_t>(
        333,
        static_cast<std::int64_t>(candidates.size())
    );

    std::vector<std::int64_t> ordered;
    ordered.reserve(candidates.size());
    ordered.push_back(candidates.front());

    std::vector<float> nearest_dist(candidates.size(), std::numeric_limits<float>::max());
    std::vector<unsigned char> selected(candidates.size(), 0);
    selected[0] = 1;

    for (std::int64_t selected_count = 1;
         selected_count < reorder_limit;
         ++selected_count) {
        const std::int64_t last = ordered.back();
        std::int64_t best_id = -1;
        float best_dist = -1.0f;

        for (std::int64_t i = 0; i < static_cast<std::int64_t>(candidates.size()); ++i) {
            if (selected[static_cast<std::size_t>(i)]) {
                continue;
            }

            nearest_dist[static_cast<std::size_t>(i)] = std::min(
                nearest_dist[static_cast<std::size_t>(i)],
                dist_sq(data, candidates[static_cast<std::size_t>(i)], last, dims)
            );

            if (nearest_dist[static_cast<std::size_t>(i)] > best_dist) {
                best_dist = nearest_dist[static_cast<std::size_t>(i)];
                best_id = i;
            }
        }

        selected[static_cast<std::size_t>(best_id)] = 1;
        ordered.push_back(candidates[static_cast<std::size_t>(best_id)]);

        if (selected_count % 100 == 0 || selected_count + 1 == reorder_limit) {
            print_progress("reorder", selected_count + 1, reorder_limit);
        }
    }

    for (std::int64_t i = 0; i < static_cast<std::int64_t>(candidates.size()); ++i) {
        if (!selected[static_cast<std::size_t>(i)]) {
            ordered.push_back(candidates[static_cast<std::size_t>(i)]);
        }
    }

    print_progress("reorder", reorder_limit, reorder_limit);

    return ordered;
}

std::vector<std::int64_t> filter_centers(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    const std::vector<std::int64_t>& centers,
    float radius_sq
) {
    std::vector<unsigned char> used(centers.size(), 0);

    for (std::int64_t i = 0; i < data.n; ++i) {
        for (std::int64_t c = 0; c < static_cast<std::int64_t>(centers.size()); ++c) {
            if (dist_sq(data, i, centers[static_cast<std::size_t>(c)], dims) <= radius_sq) {
                used[static_cast<std::size_t>(c)] = 1;
                break;
            }
        }

        if (i % 10000 == 0 || i + 1 == data.n) {
            print_progress("filter ", i + 1, data.n);
        }
    }

    std::vector<std::int64_t> filtered;
    filtered.reserve(centers.size());

    for (std::int64_t c = 0; c < static_cast<std::int64_t>(centers.size()); ++c) {
        if (used[static_cast<std::size_t>(c)]) {
            filtered.push_back(centers[static_cast<std::size_t>(c)]);
        }
    }

    return filtered;
}

} // namespace

float compute_variance_sum(
    const Dataset& data,
    const std::vector<std::int64_t>& dims
) {
    return compute_variance_sum_with_time(data, dims).variance_sum;
}

VarianceResult compute_variance_sum_with_time(
    const Dataset& data,
    const std::vector<std::int64_t>& dims
) {
    const auto start = Clock::now();
    double variance_sum = 0.0;

    for (std::int64_t dim_id = 0; dim_id < static_cast<std::int64_t>(dims.size()); ++dim_id) {
        const std::int64_t dim = dims[static_cast<std::size_t>(dim_id)];
        double mean = 0.0;

        for (std::int64_t i = 0; i < data.n; ++i) {
            mean += data.data_ptr[i * data.d + dim];
        }

        mean /= static_cast<double>(data.n);

        double var = 0.0;
        for (std::int64_t i = 0; i < data.n; ++i) {
            const double diff = data.data_ptr[i * data.d + dim] - mean;
            var += diff * diff;
        }

        variance_sum += var / static_cast<double>(data.n);
        print_progress("variance", dim_id + 1, static_cast<std::int64_t>(dims.size()));
    }

    VarianceResult result;
    result.variance_sum = static_cast<float>(variance_sum);
    result.ms = elapsed_ms(start, Clock::now());
    return result;
}

KCenterResult run_parallel_kcenter(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    float radius_sq,
    std::int64_t grain
) {
    const auto total_start = Clock::now();
    const std::int64_t block_count = (data.n + grain - 1) / grain;
    std::vector<std::vector<std::int64_t>> block_centers(static_cast<std::size_t>(block_count));

    const auto local_start = Clock::now();
    for (std::int64_t block = 0; block < block_count; ++block) {
        const std::int64_t begin = block * grain;
        const std::int64_t end = std::min(begin + grain, data.n);
        block_centers[static_cast<std::size_t>(block)] = run_block_kcenter(
            data,
            dims,
            begin,
            end,
            radius_sq
        );
        print_progress("local  ", block + 1, block_count);
    }
    const auto local_end = Clock::now();

    std::vector<std::int64_t> candidates;
    for (const auto& centers : block_centers) {
        candidates.insert(candidates.end(), centers.begin(), centers.end());
    }

    const auto reorder_start = Clock::now();
    const std::vector<std::int64_t> ordered = farthest_first_order(data, dims, candidates);
    const auto reorder_end = Clock::now();

    KCenterResult result;
    const auto filter_start = Clock::now();
    result.centers = filter_centers(data, dims, ordered, radius_sq);
    const auto filter_end = Clock::now();
    result.local_ms = elapsed_ms(local_start, local_end);
    result.reorder_ms = elapsed_ms(reorder_start, reorder_end);
    result.filter_ms = elapsed_ms(filter_start, filter_end);
    result.total_ms = elapsed_ms(total_start, Clock::now());
    return result;
}

KCenterResult run_kcenter2(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    float radius_sq
) {
    const auto start = Clock::now();
    KCenterResult result;

    if (data.n <= 0) {
        return result;
    }

    result.centers.reserve(1024);
    result.centers.push_back(0);

    for (std::int64_t i = 1; i < data.n; ++i) {
        bool covered = false;

        for (const std::int64_t center : result.centers) {
            if (dist_sq(data, i, center, dims) <= radius_sq) {
                covered = true;
                break;
            }
        }

        if (!covered) {
            result.centers.push_back(i);
        }

        if (i % 10000 == 0 || i + 1 == data.n) {
            print_progress("kcenter2", i + 1, data.n);
        }
    }

    result.total_ms = elapsed_ms(start, Clock::now());
    result.local_ms = result.total_ms;
    return result;
}
