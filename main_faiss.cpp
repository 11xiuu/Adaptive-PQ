#include "dataset.hpp"
#include "index.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

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
    if (total <= 0) return;
    const int width = 40;
    const double ratio = static_cast<double>(current) / static_cast<double>(total);
    const int filled = static_cast<int>(ratio * width);
    std::cout << '\r' << name << " [";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '.');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << '%';
    if (current >= total) std::cout << '\n';
    std::cout.flush();
}

} // namespace

int main() {
    std::cout << "loading dataset\n";
    const std::filesystem::path dataset_dir = "/home/zhaixiue/data/sift";
    const std::int64_t max_base_n = 100000;
    const Dataset base = Dataset::readFvecs((dataset_dir / "sift_base.fvecs").string(), max_base_n);
    const Dataset query = Dataset::readFvecs((dataset_dir / "sift_query.fvecs").string(), 100);
    std::cout << "base n: " << base.n << "  base d: " << base.d << '\n';
    std::cout << "query n: " << query.n << "  query d: " << query.d << '\n';

    const std::int64_t topk = 10;
    const std::int64_t ivf_list_size = static_cast<std::int64_t>(std::sqrt(base.n));
    const std::filesystem::path result_dir = "../result/" + dataset_dir.stem().string() + "_" + std::to_string(max_base_n);
    std::filesystem::create_directories(result_dir);

    // Ground truth
    std::cout << "building ground truth\n";
    std::vector<std::vector<std::int64_t>> truth(static_cast<std::size_t>(query.n));
    const auto truth_start = Clock::now();
    for (std::int64_t qi = 0; qi < query.n; ++qi) {
        truth[static_cast<std::size_t>(qi)] = exact_topk(
            base, query.data_ptr + qi * query.d, topk);
        print_progress("truth", qi + 1, query.n);
    }
    const double truth_ms = elapsed_ms(truth_start, Clock::now());
    std::cout << "truth_ms: " << truth_ms << '\n';

    const std::vector<std::int64_t> nprobes = {1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50};

    // ── FAISS IVFPQ test ──
    // Two configs: {M=8, nbits=8} and {M=16, nbits=4}
    using Config = std::pair<std::string, std::pair<std::int64_t, std::int64_t>>;
    const std::vector<Config> configs = {
        {"faiss-IVFPQ 8*8",  {8,  8}},
        {"faiss-IVFPQ 16*4", {16, 4}},
    };

    for (const auto& cfg : configs) {
        const std::string& label = cfg.first;
        const std::int64_t M = cfg.second.first;
        const std::int64_t nbits = cfg.second.second;

        std::cout << "\n=== " << label << " ===\n";
        std::cout << "M=" << M << " nbits=" << nbits << " nlist=" << ivf_list_size << '\n';

        faiss::IndexFlatL2 coarse_quantizer(base.d);
        faiss::IndexIVFPQ index(&coarse_quantizer, base.d, ivf_list_size, M, nbits);
        index.own_fields = false;
        index.cp.niter = 10;       // 减少 k-means 迭代（默认 25）
        index.cp.spherical = false; // 不做球面 k-means（更省时）
        index.pq.cp.niter = 10;    // PQ 子空间 k-means 迭代也减到 10

        auto t0 = Clock::now();
        index.train(base.n, base.data_ptr);
        index.add(base.n, base.data_ptr);
        auto t1 = Clock::now();
        const double build_ms = elapsed_ms(t0, t1);
        std::cout << "Build time: " << build_ms << " ms\n";
        std::cout << "Index size: " << index.ntotal << '\n';

        constexpr std::int64_t kRerank = 500;
        std::vector<faiss::idx_t> labels(static_cast<std::size_t>(kRerank));
        std::vector<float> pq_dists(static_cast<std::size_t>(kRerank));

        std::vector<std::pair<std::int64_t, std::pair<double, double>>> curve;

        for (std::int64_t nprobe : nprobes) {
            index.nprobe = nprobe;
            std::int64_t hit = 0;
            auto start = Clock::now();

            for (std::int64_t qi = 0; qi < query.n; ++qi) {
                const float* q = query.data_ptr + qi * query.d;

                index.search(1, q, kRerank, pq_dists.data(), labels.data());

                std::vector<std::pair<float, std::int64_t>> exact;
                exact.reserve(static_cast<std::size_t>(kRerank));
                for (std::int64_t i = 0; i < kRerank; ++i) {
                    if (labels[static_cast<std::size_t>(i)] < 0) break;
                    const float dist = l2_sq(q, base.data_ptr + labels[static_cast<std::size_t>(i)] * base.d, base.d);
                    exact.emplace_back(dist, labels[static_cast<std::size_t>(i)]);
                }

                const std::int64_t n = std::min(topk, static_cast<std::int64_t>(exact.size()));
                std::partial_sort(exact.begin(), exact.begin() + n, exact.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

                for (std::int64_t i = 0; i < n; ++i) {
                    if (std::find(truth[static_cast<std::size_t>(qi)].begin(),
                                  truth[static_cast<std::size_t>(qi)].end(),
                                  exact[static_cast<std::size_t>(i)].second) != truth[static_cast<std::size_t>(qi)].end()) {
                        ++hit;
                    }
                }

                print_progress((label + " np=" + std::to_string(nprobe)).c_str(), qi + 1, query.n);
            }

            const double total_ms = elapsed_ms(start, Clock::now());
            const double avg_ms = total_ms / static_cast<double>(query.n);
            const double recall = static_cast<double>(hit) / static_cast<double>(query.n * topk);

            std::cout << std::fixed << std::setprecision(4)
                      << " nprobe=" << nprobe << "  time=" << avg_ms << " ms  recall=" << recall << '\n';

            curve.emplace_back(nprobe, std::make_pair(avg_ms, recall));
        }

        // Write JSON
        std::ostringstream fname;
        fname << label << ".json";
        std::string safe_name = fname.str();
        for (auto& ch : safe_name) if (ch == ' ') ch = '_';
        const std::filesystem::path json_path = result_dir / safe_name;
        std::ofstream json(json_path);
        json << "{\n";
        json << "  \"name\": \"" << label << "\",\n";
        json << "  \"build_ms\": " << build_ms << ",\n";
        json << "  \"average_bits\": " << (M * nbits) << ",\n";
        json << "  \"topk\": " << topk << ",\n";
        json << "  \"base_n\": " << base.n << ",\n";
        json << "  \"base_d\": " << base.d << ",\n";
        json << "  \"query_n\": " << query.n << ",\n";
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

    return 0;
}
