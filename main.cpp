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
#include <utility>
#include <vector>
#include <cmath>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
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

struct FaissPQEval {
    std::int64_t candidate_count;
    double avg_time_ms;
    double recall;
};

std::vector<FaissPQEval> evaluate_faiss_pq(
    const Dataset& base,
    const Dataset& query,
    const std::vector<std::vector<std::int64_t>>& truth,
    std::int64_t topk,
    const std::vector<std::int64_t>& candidate_counts,
    std::int64_t M,
    std::int64_t nbits
) {
    const std::int64_t d = base.d;
    const std::int64_t nq = query.n;

    std::cout << "\n=== FAISS PQ Test ===\n";
    std::cout << "Config: M=" << M << " nbits=" << nbits << '\n';

    faiss::IndexPQ pq_index(d, M, nbits, faiss::METRIC_L2);

    auto t0 = Clock::now();
    pq_index.train(base.n, base.data_ptr);
    pq_index.add(base.n, base.data_ptr);
    auto t1 = Clock::now();
    std::cout << "Build time: " << elapsed_ms(t0, t1) << " ms\n";
    std::cout << "Index size: " << pq_index.ntotal << '\n';

    std::vector<FaissPQEval> results;

    for (std::int64_t cand : candidate_counts) {
        std::int64_t hit = 0;
        auto start = Clock::now();

        std::vector<faiss::idx_t> labels(static_cast<std::size_t>(cand));
        std::vector<float> pq_dists(static_cast<std::size_t>(cand));

        for (std::int64_t qi = 0; qi < nq; ++qi) {
            const float* q = query.data_ptr + qi * d;

            // Step 1: FAISS PQ approximate search — get cand candidates
            pq_index.search(1, q, cand, pq_dists.data(), labels.data());

            // Step 2: re-rank with exact L2
            std::vector<std::pair<float, std::int64_t>> exact;
            exact.reserve(static_cast<std::size_t>(cand));
            for (std::int64_t i = 0; i < cand; ++i) {
                if (labels[static_cast<std::size_t>(i)] < 0) break;
                const float dist = l2_sq(q, base.data_ptr + labels[static_cast<std::size_t>(i)] * d, d);
                exact.emplace_back(dist, labels[static_cast<std::size_t>(i)]);
            }

            // Step 3: sort by exact distance, take top-k
            const std::int64_t n = std::min(topk, static_cast<std::int64_t>(exact.size()));
            std::partial_sort(exact.begin(), exact.begin() + n, exact.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            // Step 4: recall vs ground truth
            for (std::int64_t i = 0; i < n; ++i) {
                if (std::find(truth[static_cast<std::size_t>(qi)].begin(),
                              truth[static_cast<std::size_t>(qi)].end(),
                              exact[static_cast<std::size_t>(i)].second) != truth[static_cast<std::size_t>(qi)].end()) {
                    ++hit;
                }
            }

            print_progress(("PQ k=" + std::to_string(cand)).c_str(), qi + 1, nq);
        }

        const double total_ms = elapsed_ms(start, Clock::now());
        const double avg_ms = total_ms / static_cast<double>(nq);
        const double recall = static_cast<double>(hit) / static_cast<double>(nq * topk);

        std::cout << std::fixed << std::setprecision(4)
                  << "cand=" << cand << "  time=" << avg_ms << " ms  recall=" << recall << '\n';

        results.push_back({cand, avg_ms, recall});
    }

    return results;
}

} // namespace

int main() {
    std::cout << "building load dataset\n";
    const std::filesystem::path dataset_dir = "/home/zhaixiue/data/sift";//sift_base.fvecs,sift_query.fvecs
    const std::int64_t max_base_n = 1000000;
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
    const std::filesystem::path result_dir = "../result";
    std::filesystem::create_directories(result_dir);
    const std::filesystem::path json_path = result_dir / "benchmark_results.json";
    const std::vector<BenchmarkConfig> configs = {
        {"8x8bit", make_uniform_config(index_base.d, 8, 8), {0.6f}, 0},
        {"16x4bit", make_uniform_config(index_base.d, 16, 4), {0.6f}, 0},
        {"kcenter8", {}, {1.0f, 0.8f, 0.6f}, 8},
        {"kcenter16", {}, {1.0f, 0.8f, 0.6f}, 16}
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
    std::ofstream json(json_path);

    json << "{\n";
    json << "  \"dataset_dir\": \"" << dataset_dir.string() << "\",\n";
    json << "  \"base_n\": " << index_base.n << ",\n";
    json << "  \"base_d\": " << index_base.d << ",\n";
    json << "  \"query_n\": " << index_query.n << ",\n";
    json << "  \"query_d\": " << index_query.d << ",\n";
    json << "  \"ivf_list_size\": " << ivf_list_size << ",\n";
    json << "  \"adaptive_radius_scale\": " << adaptive_radius_scale << ",\n";
    json << "  \"topk\": " << topk << ",\n";
    json << "  \"truth_ms\": " << truth_ms << ",\n";
    json << "  \"configs\": [\n";

    for (std::size_t cfg_id = 0; cfg_id < configs.size(); ++cfg_id) {
        const auto& cfg = configs[cfg_id];
        Index index = cfg.adaptive_target > 0
            ? Index(cfg.adaptive_target, ivf_list_size, adaptive_radius_scale)
            : Index(cfg.pq, ivf_list_size);

        std::cout << "building index config=" << cfg.name << '\n';
        const auto build_start = Clock::now();
        index.build(index_base);
        const double build_ms = elapsed_ms(build_start, Clock::now());
        const double average_bits = index.average_bits();
        const std::filesystem::path config_dump_path =
            result_dir / ("list_configs_" + std::string(cfg.name) + ".json");
        index.export_list_configs(config_dump_path.string(), 5);

        std::cout << "\n";
        std::cout << "config:      " << cfg.name << '\n';
        std::cout << "subspaces:   "
                  << (cfg.adaptive_target > 0 ? cfg.adaptive_target : static_cast<std::int64_t>(cfg.pq.size()))
                  << '\n';
        std::cout << "avg_bits:    " << average_bits << '\n';
        std::cout << "config_json: " << config_dump_path << '\n';
        std::cout << "build_ms:    " << build_ms << '\n';
        std::cout << "alpha,nprobe,time_ms,recall@" << topk
                  << ",avg_bits,exact_per_query,skipped_per_query,exact_ratio,skipped_ratio\n";

        json << "    {\n";
        json << "      \"name\": \"" << cfg.name << "\",\n";
        json << "      \"subspaces\": "
             << (cfg.adaptive_target > 0 ? cfg.adaptive_target : static_cast<std::int64_t>(cfg.pq.size()))
             << ",\n";
        json << "      \"average_bits\": " << average_bits << ",\n";
        json << "      \"build_ms\": " << build_ms << ",\n";
        json << "      \"results\": [\n";

        std::int64_t json_row_id = 0;
        const std::int64_t json_row_count =
            static_cast<std::int64_t>(cfg.alphas.size() * nprobes.size());

        for (const float alpha : cfg.alphas) {
            for (const std::int64_t nprobe : nprobes) {
                std::int64_t hit = 0;
                std::int64_t exact_count = 0;
                std::int64_t skipped_count = 0;
                std::int64_t scanned_count = 0;
                std::cout << "building query benchmark config=" << cfg.name
                          << " alpha=" << alpha
                          << " nprobe=" << nprobe << '\n';
                const auto start = Clock::now();

                for (std::int64_t qi = 0; qi < index_query.n; ++qi) {
                    const QueryResult result = index.query_refine(
                        index_query.data_ptr + qi * index_query.d,
                        nprobe,
                        topk,
                        alpha
                    );
                    exact_count += result.exact_count;
                    skipped_count += result.skipped_count;
                    scanned_count += result.scanned_count;

                    const std::int64_t result_topk = std::min(
                        topk,
                        static_cast<std::int64_t>(result.results.size())
                    );

                    for (std::int64_t i = 0; i < result_topk; ++i) {
                        const std::int64_t id = result.results[static_cast<std::size_t>(i)].id;
                        const auto& exact = truth[static_cast<std::size_t>(qi)];

                        if (std::find(exact.begin(), exact.end(), id) != exact.end()) {
                            ++hit;
                        }
                    }
                }

                const double total_ms = elapsed_ms(start, Clock::now());
                const double recall = static_cast<double>(hit) /
                    static_cast<double>(index_query.n * topk);
                const double avg_ms = total_ms / static_cast<double>(index_query.n);
                const double exact_per_query = static_cast<double>(exact_count) /
                    static_cast<double>(index_query.n);
                const double skipped_per_query = static_cast<double>(skipped_count) /
                    static_cast<double>(index_query.n);
                const double scanned_per_query = static_cast<double>(scanned_count) /
                    static_cast<double>(index_query.n);
                const double exact_ratio = scanned_count == 0
                    ? 0.0
                    : static_cast<double>(exact_count) / static_cast<double>(scanned_count);
                const double skipped_ratio = scanned_count == 0
                    ? 0.0
                    : static_cast<double>(skipped_count) / static_cast<double>(scanned_count);

                std::cout << std::fixed << std::setprecision(1) << alpha << ','
                          << nprobe << ','
                          << std::fixed << std::setprecision(4) << avg_ms << ','
                          << std::fixed << std::setprecision(6) << recall << ','
                          << std::fixed << std::setprecision(2) << average_bits << ','
                          << std::fixed << std::setprecision(1) << exact_per_query << ','
                          << std::fixed << std::setprecision(1) << skipped_per_query << ','
                          << std::fixed << std::setprecision(6) << exact_ratio << ','
                          << std::fixed << std::setprecision(6) << skipped_ratio << '\n';

                json << "        {\"alpha\": " << alpha
                     << ", \"nprobe\": " << nprobe
                     << ", \"time_ms\": " << avg_ms
                     << ", \"recall\": " << recall
                     << ", \"average_bits\": " << average_bits
                     << ", \"exact_per_query\": " << exact_per_query
                     << ", \"skipped_per_query\": " << skipped_per_query
                     << ", \"scanned_per_query\": " << scanned_per_query
                     << ", \"exact_ratio\": " << exact_ratio
                     << ", \"skipped_ratio\": " << skipped_ratio
                     << "}";
                ++json_row_id;
                if (json_row_id < json_row_count) {
                    json << ",";
                }
                json << "\n";
            }
        }

        json << "      ]\n";
        json << "    }";
        if (cfg_id + 1 < configs.size()) {
            json << ",";
        }
        json << "\n";
    }

    std::cout << "building write json\n";
    json << "  ]\n";
    json << "}\n";
    std::cout << "\njson:        " << json_path << '\n';

    // ── FAISS PQ test ──
    const std::vector<std::int64_t> candidate_counts = {100, 300, 500};
    const auto faiss_results = evaluate_faiss_pq(
        index_base, index_query, truth, topk, candidate_counts, 8, 8
    );

    {
        const std::filesystem::path csv_path = result_dir / "faiss_pq_results.csv";
        std::ofstream csv(csv_path);
        csv << "candidate_count,avg_time_ms,recall\n";
        for (const auto& r : faiss_results) {
            csv << r.candidate_count << ","
                << std::fixed << std::setprecision(4) << r.avg_time_ms << ","
                << std::fixed << std::setprecision(6) << r.recall << "\n";
        }
        std::cout << "faiss_csv:   " << csv_path << '\n';
    }

    return 0;
}
