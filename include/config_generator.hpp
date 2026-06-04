#pragma once

#include "dataset.hpp"
#include "index.hpp"

#include <cstdint>
#include <vector>

std::vector<PQSubspaceConfig> make_kcenter_merge_config(
    const Dataset& data,
    std::int64_t target_subspace_count,
    float radius_sq
);

std::vector<PQSubspaceConfig> make_kcenter_merge_config(
    const Dataset& data,
    const std::vector<std::int64_t>& ids,
    std::int64_t target_subspace_count,
    float radius_sq
);
