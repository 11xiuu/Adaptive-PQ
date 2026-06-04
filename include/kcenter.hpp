#pragma once

#include "dataset.hpp"

#include <cstdint>
#include <vector>

struct KCenterResult {
    std::vector<std::int64_t> centers;
    double local_ms = 0.0;
    double reorder_ms = 0.0;
    double filter_ms = 0.0;
    double total_ms = 0.0;
};

struct VarianceResult {
    float variance_sum = 0.0f;
    double ms = 0.0;
};

float compute_variance_sum(
    const Dataset& data,
    const std::vector<std::int64_t>& dims
);

VarianceResult compute_variance_sum_with_time(
    const Dataset& data,
    const std::vector<std::int64_t>& dims
);

KCenterResult run_parallel_kcenter(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    float radius_sq,
    std::int64_t grain = 10000
);

KCenterResult run_kcenter2(
    const Dataset& data,
    const std::vector<std::int64_t>& dims,
    float radius_sq
);
