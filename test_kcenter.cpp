#include "pq.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::high_resolution_clock;

constexpr int MAX_LEVELS = 3;

// k-center 内部强行切片并行粒度：0.01M
static constexpr std::size_t KCENTER_PARALLEL_GRAIN = 10000;

// 第一层 groups 并行分发粒度
static constexpr std::size_t GROUP_PARALLEL_GRAIN = 4;

static constexpr std::size_t INVALID_ID =
    std::numeric_limits<std::size_t>::max();

#ifdef _OPENMP
int omp_threads() {
    return std::max(1, omp_get_max_threads());
}
#else
int omp_threads() {
    return 1;
}
#endif

std::uint32_t mix_seed(std::uint32_t seed, std::size_t x) {
    return seed * 1664525u + 1013904223u
           + static_cast<std::uint32_t>(x * 1000003ull);
}

struct MetricView {
    enum class Mode {
        D1,
        D2,
        D10,
        FULL,
        GENERIC
    };

    const std::vector<int>& dims;
    Mode mode = Mode::GENERIC;
    std::size_t full_dim = 0;

    explicit MetricView(const std::vector<int>& dims_, std::size_t data_dim)
        : dims(dims_), full_dim(data_dim) {
        if (dims.size() == 1) {
            mode = Mode::D1;
        } else if (dims.size() == 2) {
            mode = Mode::D2;
        } else if (dims.size() == 10) {
            mode = Mode::D10;
        } else if (
            dims.size() == data_dim &&
            !dims.empty() &&
            dims.front() == 0 &&
            dims.back() == static_cast<int>(data_dim - 1)
        ) {
            mode = Mode::FULL;
        } else {
            mode = Mode::GENERIC;
        }
    }

    inline float dist_sq(
        const pq::Dataset& data,
        std::size_t a,
        std::size_t b
    ) const {
        const auto& x = data[a];
        const auto& y = data[b];

        if (mode == Mode::D1) {
            const int d0 = dims[0];
            const float t0 = x[d0] - y[d0];
            return t0 * t0;
        }

        if (mode == Mode::D2) {
            const int d0 = dims[0];
            const int d1 = dims[1];

            const float t0 = x[d0] - y[d0];
            const float t1 = x[d1] - y[d1];

            return t0 * t0 + t1 * t1;
        }

        if (mode == Mode::D10) {
            float s = 0.0f;

            for (int i = 0; i < 10; ++i) {
                const int d = dims[static_cast<std::size_t>(i)];
                const float t = x[d] - y[d];
                s += t * t;
            }

            return s;
        }

        if (mode == Mode::FULL) {
            float s = 0.0f;

            for (std::size_t d = 0; d < full_dim; ++d) {
                const float t = x[d] - y[d];
                s += t * t;
            }

            return s;
        }

        float s = 0.0f;

        for (int d : dims) {
            const float t = x[d] - y[d];
            s += t * t;
        }

        return s;
    }
};

struct LocalKCenterPartition {
    std::vector<std::size_t> centers;
    std::vector<std::vector<std::size_t>> groups;
    double ms = 0.0;
};

struct HierKCenterResult {
    std::vector<std::size_t> leaf_centers;

    std::size_t k = 0;
    double ms = 0.0;

    std::array<std::size_t, MAX_LEVELS> level_k{};
    std::array<double, MAX_LEVELS> level_ms{};
};

void merge_result_sum_k_max_time(
    HierKCenterResult& dst,
    const HierKCenterResult& src
) {
    for (int lv = 0; lv < MAX_LEVELS; ++lv) {
        dst.level_k[lv] += src.level_k[lv];
        dst.level_ms[lv] = std::max(dst.level_ms[lv], src.level_ms[lv]);
    }

    dst.leaf_centers.insert(
        dst.leaf_centers.end(),
        src.leaf_centers.begin(),
        src.leaf_centers.end()
    );
}

// 对 points[begin, end) 顺序执行 online k-center。
// 注意：这是一个局部切片，不会考虑切片外的 centers。
LocalKCenterPartition run_online_kcenter_range(
    const pq::Dataset& data,
    const MetricView& metric,
    const std::vector<std::size_t>& points,
    std::size_t begin,
    std::size_t end,
    double radius_sq,
    bool build_groups
) {
    LocalKCenterPartition ret;

    if (begin >= end) {
        return ret;
    }

    const std::size_t m = end - begin;

    ret.centers.reserve(std::min<std::size_t>(64, m));
    if (build_groups) {
        ret.groups.reserve(std::min<std::size_t>(64, m));
    }

    ret.centers.push_back(points[begin]);

    if (build_groups) {
        ret.groups.emplace_back();
        ret.groups.back().push_back(points[begin]);
    }

    for (std::size_t j = begin + 1; j < end; ++j) {
        const std::size_t x = points[j];

        bool covered = false;
        std::size_t hit_center_id = 0;

        for (std::size_t c = 0; c < ret.centers.size(); ++c) {
            const float d2 = metric.dist_sq(
                data,
                x,
                ret.centers[c]
            );

            if (d2 <= radius_sq) {
                covered = true;
                hit_center_id = c;
                break;
            }
        }

        if (covered) {
            if (build_groups) {
                ret.groups[hit_center_id].push_back(x);
            }
        } else {
            ret.centers.push_back(x);

            if (build_groups) {
                ret.groups.emplace_back();
                ret.groups.back().push_back(x);
            }
        }
    }

    return ret;
}

// 新增步骤：
// candidate centers 合并后，对 points 并行做 first-cover assign。
// 然后删除没有被任何点命中的 center。
// assignment 同时用于重建 groups。
LocalKCenterPartition assign_and_prune_centers(
    const pq::Dataset& data,
    const MetricView& metric,
    const std::vector<std::size_t>& points,
    const std::vector<std::size_t>& candidate_centers,
    double radius_sq,
    bool build_groups,
    std::size_t parallel_grain
) {
    LocalKCenterPartition ret;

    const std::size_t n = points.size();
    const std::size_t k = candidate_centers.size();

    if (n == 0 || k == 0) {
        return ret;
    }

    std::vector<std::size_t> assignment(n, INVALID_ID);

    const std::size_t grain = std::max<std::size_t>(1, parallel_grain);
    const std::size_t n_blocks = (n + grain - 1) / grain;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t bi = 0;
         bi < static_cast<std::ptrdiff_t>(n_blocks);
         ++bi)
    {
        const std::size_t block_id = static_cast<std::size_t>(bi);
        const std::size_t begin = block_id * grain;
        const std::size_t end = std::min(begin + grain, n);

        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t x = points[i];

            for (std::size_t c = 0; c < k; ++c) {
                const float d2 = metric.dist_sq(
                    data,
                    x,
                    candidate_centers[c]
                );

                if (d2 <= radius_sq) {
                    assignment[i] = c;
                    break;
                }
            }
        }
    }

    std::vector<unsigned char> used(k, 0);

    for (std::size_t i = 0; i < n; ++i) {
        if (assignment[i] != INVALID_ID) {
            used[assignment[i]] = 1;
        }
    }

    std::vector<std::size_t> old_to_new(k, INVALID_ID);

    std::size_t kept = 0;
    for (std::size_t c = 0; c < k; ++c) {
        if (used[c]) {
            ++kept;
        }
    }

    ret.centers.reserve(kept);

    if (build_groups) {
        ret.groups.reserve(kept);
    }

    for (std::size_t c = 0; c < k; ++c) {
        if (!used[c]) {
            continue;
        }

        old_to_new[c] = ret.centers.size();
        ret.centers.push_back(candidate_centers[c]);

        if (build_groups) {
            ret.groups.emplace_back();
        }
    }

    if (build_groups) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t old_c = assignment[i];

            if (old_c != INVALID_ID) {
                const std::size_t new_c = old_to_new[old_c];
                ret.groups[new_c].push_back(points[i]);
            } else {
                // 理论上不会发生，因为 candidate centers 来自每个分片的 R-cover。
                ret.centers.push_back(points[i]);
                ret.groups.emplace_back();
                ret.groups.back().push_back(points[i]);
            }
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            if (assignment[i] == INVALID_ID) {
                // 理论兜底：未被任何 candidate center cover 的点自己成为 center。
                ret.centers.push_back(points[i]);
            }
        }
    }

    return ret;
}

// Online radius-net / streaming k-center。
// 如果 points.size() > kcenter_parallel_grain，则强行切片并行。
// 并行切片后，增加一次 assign_and_prune_centers，减少重复 center。
LocalKCenterPartition run_kcenter_partition_local(
    const pq::Dataset& data,
    const MetricView& metric,
    const std::vector<std::size_t>& points,
    double radius_sq,
    bool build_groups,
    std::size_t kcenter_parallel_grain
) {
    const auto t0 = Clock::now();

    LocalKCenterPartition ret;

    const std::size_t m = points.size();
    if (m == 0) {
        ret.ms = std::chrono::duration<double, std::milli>(
            Clock::now() - t0
        ).count();
        return ret;
    }

    const std::size_t grain = std::max<std::size_t>(1, kcenter_parallel_grain);

#ifndef _OPENMP
    ret = run_online_kcenter_range(
        data,
        metric,
        points,
        0,
        m,
        radius_sq,
        build_groups
    );
#else
    if (m <= grain) {
        ret = run_online_kcenter_range(
            data,
            metric,
            points,
            0,
            m,
            radius_sq,
            build_groups
        );
    } else {
        const std::size_t n_blocks = (m + grain - 1) / grain;

        std::vector<LocalKCenterPartition> locals(n_blocks);

        // 原来的分片并行 k-center 保持不变。
        // 注意这里不需要 build_groups，因为后面 assign 会重新得到 groups。
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t bi = 0;
             bi < static_cast<std::ptrdiff_t>(n_blocks);
             ++bi)
        {
            const std::size_t block_id = static_cast<std::size_t>(bi);
            const std::size_t begin = block_id * grain;
            const std::size_t end = std::min(begin + grain, m);

            locals[block_id] = run_online_kcenter_range(
                data,
                metric,
                points,
                begin,
                end,
                radius_sq,
                false
            );
        }

        std::size_t total_centers = 0;

        for (const auto& local : locals) {
            total_centers += local.centers.size();
        }

        std::vector<std::size_t> candidate_centers;
        candidate_centers.reserve(total_centers);

        for (auto& local : locals) {
            candidate_centers.insert(
                candidate_centers.end(),
                local.centers.begin(),
                local.centers.end()
            );
        }

        // 只增加这一步：first-cover assign + prune unused centers。
        ret = assign_and_prune_centers(
            data,
            metric,
            points,
            candidate_centers,
            radius_sq,
            build_groups,
            grain
        );
    }
#endif

    ret.ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0
    ).count();

    return ret;
}

void hierarchical_kcenter_recursive(
    const pq::Dataset& data,
    const MetricView& metric,
    const std::vector<std::size_t>& points,
    double radius_sq,
    double shrink_ratio,
    int depth,
    int max_depth,
    std::uint32_t seed,
    std::size_t kcenter_parallel_grain,
    std::size_t group_parallel_grain,
    HierKCenterResult& out
);

void run_first_level_children_parallel(
    const pq::Dataset& data,
    const MetricView& metric,
    const std::vector<std::vector<std::size_t>>& groups,
    double child_radius_sq,
    double shrink_ratio,
    int max_depth,
    std::uint32_t seed,
    std::size_t kcenter_parallel_grain,
    std::size_t group_parallel_grain,
    HierKCenterResult& out
) {
    const std::size_t g = groups.size();
    if (g == 0) {
        return;
    }

    const std::size_t grain = std::max<std::size_t>(1, group_parallel_grain);
    const std::size_t n_blocks = (g + grain - 1) / grain;

    std::vector<HierKCenterResult> locals(n_blocks);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t bi = 0;
         bi < static_cast<std::ptrdiff_t>(n_blocks);
         ++bi)
    {
        const std::size_t block_id = static_cast<std::size_t>(bi);
        const std::size_t begin = block_id * grain;
        const std::size_t end = std::min(begin + grain, g);

        auto& local_out = locals[block_id];

        for (std::size_t i = begin; i < end; ++i) {
            if (groups[i].empty()) {
                continue;
            }

            hierarchical_kcenter_recursive(
                data,
                metric,
                groups[i],
                child_radius_sq,
                shrink_ratio,
                1,
                max_depth,
                mix_seed(seed, i),
                kcenter_parallel_grain,
                group_parallel_grain,
                local_out
            );
        }
    }

    for (const auto& local : locals) {
        merge_result_sum_k_max_time(out, local);
    }
}

void hierarchical_kcenter_recursive(
    const pq::Dataset& data,
    const MetricView& metric,
    const std::vector<std::size_t>& points,
    double radius_sq,
    double shrink_ratio,
    int depth,
    int max_depth,
    std::uint32_t seed,
    std::size_t kcenter_parallel_grain,
    std::size_t group_parallel_grain,
    HierKCenterResult& out
) {
    (void)seed;

    if (points.empty()) {
        return;
    }

    const bool is_leaf = (depth + 1 >= max_depth);

    auto part = run_kcenter_partition_local(
        data,
        metric,
        points,
        radius_sq,
        !is_leaf,
        kcenter_parallel_grain
    );

    if (depth >= 0 && depth < MAX_LEVELS) {
        out.level_k[depth] += part.centers.size();
        out.level_ms[depth] += part.ms;
    }

    if (is_leaf) {
        out.leaf_centers.insert(
            out.leaf_centers.end(),
            part.centers.begin(),
            part.centers.end()
        );
        return;
    }

    const double child_radius_sq =
        radius_sq / (shrink_ratio * shrink_ratio);

    // 第一层 groups 做 block-level 并行。
    if (depth == 0) {
        run_first_level_children_parallel(
            data,
            metric,
            part.groups,
            child_radius_sq,
            shrink_ratio,
            max_depth,
            seed,
            kcenter_parallel_grain,
            group_parallel_grain,
            out
        );
        return;
    }

    // 后续递归层顺序分发。
    for (std::size_t i = 0; i < part.groups.size(); ++i) {
        if (part.groups[i].empty()) {
            continue;
        }

        hierarchical_kcenter_recursive(
            data,
            metric,
            part.groups[i],
            child_radius_sq,
            shrink_ratio,
            depth + 1,
            max_depth,
            mix_seed(seed, i + static_cast<std::size_t>(depth) * 100000ull),
            kcenter_parallel_grain,
            group_parallel_grain,
            out
        );
    }
}

HierKCenterResult run_hierarchical_kcenter(
    const pq::Dataset& data,
    const std::vector<int>& dims,
    double radius_sq,
    double shrink_ratio,
    int max_depth,
    std::uint32_t seed,
    std::size_t kcenter_parallel_grain,
    std::size_t group_parallel_grain
) {
    const auto t0 = Clock::now();

    MetricView metric(dims, data.front().size());

    std::vector<std::size_t> all_points(data.size());
    std::iota(all_points.begin(), all_points.end(), std::size_t(0));

    HierKCenterResult out;

    hierarchical_kcenter_recursive(
        data,
        metric,
        all_points,
        radius_sq,
        shrink_ratio,
        0,
        max_depth,
        seed,
        kcenter_parallel_grain,
        group_parallel_grain,
        out
    );

    out.k = out.leaf_centers.size();

    out.ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0
    ).count();

    return out;
}

void print_hier_result(const HierKCenterResult& r) {
    std::cout << "  hierarchical:\n";

    for (int lv = 0; lv < MAX_LEVELS; ++lv) {
        std::cout << "    L" << (lv + 1)
                  << ": k=" << r.level_k[lv]
                  << ", time(ms)=" << std::fixed << std::setprecision(1)
                  << r.level_ms[lv]
                  << "\n";
    }

    std::cout << "    total: k=" << r.k
              << ", time(ms)=" << std::fixed << std::setprecision(1)
              << r.ms
              << "\n";
}

void print_direct_result(const HierKCenterResult& r) {
    std::cout << "  direct:\n";

    std::cout << "    k=" << r.k
              << ", time(ms)=" << std::fixed << std::setprecision(1)
              << r.ms
              << "\n";
}

} // namespace

int main() {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_max_active_levels(2);
#endif

    std::cout << "Loading SIFT base dataset...\n";
    std::cout.flush();

    auto base = pq::load_fvecs("../data/sift/sift_base.fvecs");
    // base.resize(100'000);

    std::cout << "Loaded " << base.size()
              << " vectors, dim=" << base.front().size() << "\n";

    auto variances = pq::compute_dim_variances(base);
    const int D = static_cast<int>(variances.size());

    const double radius_scale = 1.0;

    double global_energy = 0.0;
    for (int d = 0; d < D; ++d) {
        global_energy += variances[static_cast<std::size_t>(d)];
    }

    double radius_sq = radius_scale * radius_scale * global_energy;
    // radius_sq /= 4.0;

    const int HIER_DEPTH = 3;
    const double SHRINK_RATIO = 1.8;

    const double finest_radius_sq =
        radius_sq / std::pow(
            SHRINK_RATIO,
            2.0 * static_cast<double>(HIER_DEPTH - 1)
        );

    struct TestCase {
        std::string name;
        std::vector<int> dims;
    };

    std::vector<TestCase> tests;

    tests.push_back({"dim[0]", {0}});
    tests.push_back({"dim[0,1]", {0, 1}});

    std::vector<int> first10;
    for (int i = 0; i < 10 && i < D; ++i) {
        first10.push_back(i);
    }
    tests.push_back({"dim[0..9]", first10});

    first10.clear();
    for (int i = 0; i < 20 && i < D; ++i) {
        first10.push_back(i);
    }
    tests.push_back({"dim[0..19]", first10});

    // std::vector<int> all_dims(D);
    // std::iota(all_dims.begin(), all_dims.end(), 0);
    // tests.push_back({
    //     "dim[0.." + std::to_string(D - 1) + "]",
    //     all_dims
    // });

    const std::size_t KCENTER_GRAIN = KCENTER_PARALLEL_GRAIN;
    const std::size_t GROUP_GRAIN = GROUP_PARALLEL_GRAIN;

    std::cout << "\n";
    std::cout << "init_radius=" << std::sqrt(radius_sq)
              << ", finest_radius=" << std::sqrt(finest_radius_sq)
              << ", shrink_ratio=" << SHRINK_RATIO
              << ", kcenter_grain=" << KCENTER_GRAIN
              << ", group_grain=" << GROUP_GRAIN
#ifdef _OPENMP
              << ", omp_threads=" << omp_threads()
#else
              << ", omp=off"
#endif
              << "\n";

    std::cout << "============================================================\n";

    for (const auto& tc : tests) {
        std::cout << "\n";
        std::cout << "Subspace: " << tc.name
                  << ", dims=" << tc.dims.size()
                  << "\n";

        auto r_hier = run_hierarchical_kcenter(
            base,
            tc.dims,
            radius_sq,
            SHRINK_RATIO,
            HIER_DEPTH,
            123,
            KCENTER_GRAIN,
            GROUP_GRAIN
        );

        auto r_direct = run_hierarchical_kcenter(
            base,
            tc.dims,
            finest_radius_sq,
            SHRINK_RATIO,
            1,
            123,
            KCENTER_GRAIN,
            GROUP_GRAIN
        );

        print_hier_result(r_hier);
        print_direct_result(r_direct);

        std::cout.flush();
    }

    std::cout << "\nDone.\n";

    return 0;
}