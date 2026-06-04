#include "config_generator.hpp"

#include <algorithm>
#include <limits>

namespace {

struct SubspaceState {
    std::vector<std::int64_t> dims;
    std::int64_t center_count = 0;
};

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

std::int64_t kcenter_count(
    const Dataset& data,
    const std::vector<std::int64_t>& ids,
    const std::vector<std::int64_t>& dims,
    float radius_sq
) {
    if (ids.empty()) {
        return 0;
    }

    std::vector<std::int64_t> centers;
    centers.reserve(1024);
    centers.push_back(ids.front());

    for (std::int64_t pos = 1; pos < static_cast<std::int64_t>(ids.size()); ++pos) {
        const std::int64_t id = ids[static_cast<std::size_t>(pos)];
        bool covered = false;

        for (const std::int64_t center : centers) {
            if (dist_sq(data, id, center, dims) <= radius_sq) {
                covered = true;
                break;
            }
        }

        if (!covered) {
            centers.push_back(id);
        }
    }

    return static_cast<std::int64_t>(centers.size());
}

std::vector<std::int64_t> merged_dims(
    const std::vector<std::int64_t>& a,
    const std::vector<std::int64_t>& b
) {
    std::vector<std::int64_t> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

std::vector<PQSubspaceConfig> make_kcenter_merge_config(
    const Dataset& data,
    std::int64_t target_subspace_count,
    float radius_sq
) {
    std::vector<std::int64_t> ids(static_cast<std::size_t>(data.n));
    for (std::int64_t i = 0; i < data.n; ++i) {
        ids[static_cast<std::size_t>(i)] = i;
    }
    return make_kcenter_merge_config(data, ids, target_subspace_count, radius_sq);
}

std::vector<PQSubspaceConfig> make_kcenter_merge_config(
    const Dataset& data,
    const std::vector<std::int64_t>& ids,
    std::int64_t target_subspace_count,
    float radius_sq
) {
    std::vector<SubspaceState> active;
    active.reserve(static_cast<std::size_t>(data.d));

    for (std::int64_t dim = 0; dim < data.d; ++dim) {
        SubspaceState state;
        state.dims.push_back(dim);
        state.center_count = kcenter_count(data, ids, state.dims, radius_sq);
        active.push_back(std::move(state));
    }

    while (static_cast<std::int64_t>(active.size()) > target_subspace_count) {
        if (active.size() <= 1) {
            break;
        }

        const auto min_it = std::min_element(
            active.begin(),
            active.end(),
            [](const SubspaceState& a, const SubspaceState& b) {
                return a.center_count < b.center_count;
            }
        );
        const std::int64_t src = static_cast<std::int64_t>(std::distance(active.begin(), min_it));

        std::int64_t best_dst = -1;
        std::int64_t best_k = 0;
        std::int64_t best_growth = std::numeric_limits<std::int64_t>::max();
        std::vector<std::int64_t> best_dims;

        for (std::int64_t dst = 0; dst < static_cast<std::int64_t>(active.size()); ++dst) {
            if (dst == src) {
                continue;
            }

            std::vector<std::int64_t> dims = merged_dims(
                active[static_cast<std::size_t>(src)].dims,
                active[static_cast<std::size_t>(dst)].dims
            );
            const std::int64_t merged_k = kcenter_count(data, ids, dims, radius_sq);
            const std::int64_t src_k = active[static_cast<std::size_t>(src)].center_count;
            const std::int64_t growth = merged_k - src_k;

            if (growth < best_growth) {
                best_growth = growth;
                best_dst = dst;
                best_k = merged_k;
                best_dims = std::move(dims);
            }
        }

        const std::int64_t keep = std::min(src, best_dst);
        const std::int64_t erase = std::max(src, best_dst);
        active[static_cast<std::size_t>(keep)].dims = std::move(best_dims);
        active[static_cast<std::size_t>(keep)].center_count = best_k;
        active.erase(active.begin() + erase);
    }

    std::vector<PQSubspaceConfig> config;
    config.reserve(active.size());

    for (auto& subspace : active) {
        std::sort(subspace.dims.begin(), subspace.dims.end());
        PQSubspaceConfig item;
        item.dims = std::move(subspace.dims);
        item.codebook_size = std::max<std::int64_t>(1, subspace.center_count);
        config.push_back(std::move(item));
    }

    return config;
}
