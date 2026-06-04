#include "kcenter_benchmark.hpp"

#include "kcenter.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

void run_kcenter_benchmark(const Dataset& base) {
    std::vector<std::int64_t> dims;
    for (std::int64_t i = 0; i < 5; ++i) {
        dims.push_back(i);
    }

    const VarianceResult variance = compute_variance_sum_with_time(base, dims);
    const float radius_sq = 0.5f * variance.variance_sum;
    const KCenterResult result = run_kcenter2(base, dims, radius_sq);

    std::cout << "variance_ms: " << variance.ms << '\n';
    std::cout << "radius_sq:   " << radius_sq << '\n';
    std::cout << "kcenter2 k:  " << result.centers.size() << '\n';
    std::cout << "kcenter2_ms: " << result.total_ms << '\n';
}
