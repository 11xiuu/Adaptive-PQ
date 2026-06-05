#include "config_generator.hpp"
#include "dataset.hpp"
#include "index.hpp"
#include "kcenter_benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
namespace {

using Clock = std::chrono::steady_clock;

float l2_sq(const float* a, const float* b, std::int64_t d) {
    float sum = 0.0f;
    for (std::int64_t i = 0; i < d; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

struct BenchmarkConfig {
    const char* name = "";
    std::vector<PQSubspaceConfig> pq;
    std::vector<float> alphas;
    std::int64_t adaptive_target = 0;
    bool is_residual = false;
};

std::vector<PQSubspaceConfig> make_uniform_config(
    std::int64_t d,
    std::int64_t subspace_count,
    std::int64_t bits
) {
    std::vector<PQSubspaceConfig> config;
    const std::int64_t codebook_size = std::int64_t{1} << bits;

    for (std::int64_t s = 0; s < subspace_count; ++s) {
        PQSubspaceConfig item;
        item.codebook_size = codebook_size;

        const std::int64_t begin = s * d / subspace_count;
        const std::int64_t end = (s + 1) * d / subspace_count;
        for (std::int64_t dim = begin; dim < end; ++dim) {
            item.dims.push_back(dim);
        }

        config.push_back(std::move(item));
    }

    return config;
}

std::vector<std::int64_t> exact_topk(const Dataset& base, const float* q, std::int64_t topk) {
    std::vector<SearchResult> dist(static_cast<std::size_t>(base.n));

    for (std::int64_t i = 0; i < base.n; ++i) {
        dist[static_cast<std::size_t>(i)] = {
            i,
            l2_sq(q, base.data_ptr + i * base.d, base.d)
        };
    }

    std::partial_sort(
        dist.begin(),
        dist.begin() + topk,
        dist.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    std::vector<std::int64_t> ids;
    ids.reserve(static_cast<std::size_t>(topk));
    for (std::int64_t i = 0; i < topk; ++i) {
        ids.push_back(dist[static_cast<std::size_t>(i)].id);
    }
    return ids;
}

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

} // namespace

int main() {
    std::cout << "building load dataset\n";
    const std::filesystem::path dataset_dir = "/home/zhaixiue/data/sift";//sift_base.fvecs,sift_query.fvecs
    const std::int64_t max_base_n = 100000;
    // const Dataset base((dataset_dir / "data.tensor").string(), max_base_n);
    // const Dataset query((dataset_dir / "query.tensor").string(), 100);
    const Dataset base = Dataset::readFvecs((dataset_dir / "sift_base.fvecs").string(), max_base_n);
    const Dataset query = Dataset::readFvecs((dataset_dir / "sift_query.fvecs").string(), 100);
    std::cout << "dataset_dir: " << dataset_dir << '\n';
    std::cout << "base n:      " << base.n << '\n';
    std::cout << "base d:      " << base.d << '\n';
    std::cout << "query n:     " << query.n << '\n';
    std::cout << "query d:     " << query.d << '\n';

    // run_kcenter_benchmark(base);

    const Dataset& index_base = base;
    const Dataset& index_query = query;

    const std::int64_t topk = 10;
    const std::int64_t ivf_list_size = std::sqrt(index_base.n);
    const float adaptive_radius_scale = 0.03f;
    const std::filesystem::path result_dir = "../result/" + dataset_dir.stem().string() + "_" + std::to_string(max_base_n);
    std::filesystem::create_directories(result_dir);
    const std::vector<BenchmarkConfig> configs = {
        {"8x8bit-residual", make_uniform_config(index_base.d, 8, 8), {0.8f}, 0, true},
        {"16x4bit-residual", make_uniform_config(index_base.d, 16, 4), {0.8f}, 0, true},
        {"kcenter8-residual", {}, {1.0f, 0.8f}, 8, true},
        {"kcenter16-residual", {}, {1.0f, 0.8f}, 16, true}
    };

    std::cout << "building ground truth\n";
    std::vector<std::vector<std::int64_t>> truth(static_cast<std::size_t>(index_query.n));
    const auto truth_start = Clock::now();
    for (std::int64_t qi = 0; qi < index_query.n; ++qi) {
        truth[static_cast<std::size_t>(qi)] = exact_topk(
            index_base,
            index_query.data_ptr + qi * index_query.d,
            topk
        );
        print_progress("truth  ", qi + 1, index_query.n);
    }
    const double truth_ms = elapsed_ms(truth_start, Clock::now());

    std::cout << "ivf lists:   " << ivf_list_size << '\n';
    std::cout << "topk:        " << topk << '\n';
    std::cout << "truth_ms:    " << truth_ms << '\n';

    const std::vector<std::int64_t> nprobes = {1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50};

    for (const auto& cfg : configs) {
        Index index = cfg.adaptive_target > 0
            ? Index(cfg.adaptive_target, ivf_list_size, adaptive_radius_scale)
            : Index(cfg.pq, ivf_list_size);

        std::cout << "\nbuilding index config=" << cfg.name << '\n';
        index.set_residual(cfg.is_residual);
        const auto build_start = Clock::now();
        index.build(index_base);
        const double build_ms = elapsed_ms(build_start, Clock::now());
        const double avg_bits = index.average_bits();

        std::cout << "build_ms:    " << build_ms << '\n';
        std::cout << "avg_bits:    " << avg_bits << '\n';

        for (const float alpha : cfg.alphas) {
            std::cout << "\n  alpha=" << alpha << '\n';

            std::vector<std::pair<std::int64_t, std::pair<double, double>>> curve;
            curve.reserve(nprobes.size());

            for (const std::int64_t nprobe : nprobes) {
                std::int64_t hit = 0;
                const auto start = Clock::now();

                for (std::int64_t qi = 0; qi < index_query.n; ++qi) {
                    const QueryResult result = index.query_refine(
                        index_query.data_ptr + qi * index_query.d,
                        nprobe, topk, alpha);
                    const std::int64_t result_topk = std::min(
                        topk, static_cast<std::int64_t>(result.results.size()));
                    for (std::int64_t i = 0; i < result_topk; ++i) {
                        const std::int64_t id = result.results[static_cast<std::size_t>(i)].id;
                        if (std::find(truth[static_cast<std::size_t>(qi)].begin(),
                                      truth[static_cast<std::size_t>(qi)].end(), id) != truth[static_cast<std::size_t>(qi)].end()) {
                            ++hit;
                        }
                    }
                }

                const double total_ms = elapsed_ms(start, Clock::now());
                const double avg_ms = total_ms / static_cast<double>(index_query.n);
                const double recall = static_cast<double>(hit) /
                    static_cast<double>(index_query.n * topk);

                std::cout << std::fixed << std::setprecision(4)
                          << "    nprobe=" << nprobe << "  time=" << avg_ms << " ms  recall=" << recall << '\n';

                curve.emplace_back(nprobe, std::make_pair(avg_ms, recall));
            }

            // Write per-(config, alpha) JSON
            std::ostringstream fname;
            fname << cfg.name << "_a" << std::setprecision(1) << alpha << ".json";
            const std::filesystem::path json_path = result_dir / fname.str();
            std::ofstream json(json_path);
            json << "{\n";
            json << "  \"name\": \"" << cfg.name << "\",\n";
            json << "  \"alpha\": " << alpha << ",\n";
            json << "  \"build_ms\": " << build_ms << ",\n";
            json << "  \"average_bits\": " << avg_bits << ",\n";
            json << "  \"topk\": " << topk << ",\n";
            json << "  \"base_n\": " << index_base.n << ",\n";
            json << "  \"base_d\": " << index_base.d << ",\n";
            json << "  \"query_n\": " << index_query.n << ",\n";
            json << "  \"results\": [\n";
            for (std::size_t i = 0; i < curve.size(); ++i) {
                if (i > 0) json << ",\n";
                json << "    {\"nprobe\": " << curve[i].first
                     << ", \"time_ms\": " << std::fixed << std::setprecision(4) << curve[i].second.first
                     << ", \"recall\": " << std::setprecision(6) << curve[i].second.second
                     << "}";
            }
            json << "\n  ]\n}\n";
            std::cout << "  saved: " << json_path.filename() << '\n';
        }
    }

    return 0;
}
