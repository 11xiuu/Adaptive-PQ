#include "index.hpp"

#include "config_generator.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

constexpr const char* kIVFCacheDir = "./data/index_buffer/PQ";

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

void print_list_progress(std::int64_t current, std::int64_t total) {
    if (total <= 0) {
        return;
    }

    const int width = 40;
    const double ratio = static_cast<double>(current) / static_cast<double>(total);
    const int filled = static_cast<int>(ratio * width);

    std::cout << '\r' << "pq list " << current << "/" << total << " [";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '.');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << '%';

    if (current >= total) {
        std::cout << '\n';
    }
    std::cout.flush();
}

void print_kmeans_iter_progress(
    const char* name,
    std::int64_t iter,
    std::int64_t iters,
    std::int64_t current,
    std::int64_t total
) {
    if (total <= 0) {
        return;
    }

    const int width = 40;
    const double ratio = static_cast<double>(current) / static_cast<double>(total);
    const int filled = static_cast<int>(ratio * width);

    std::cout << '\r' << name << " iter " << iter << "/" << iters << " [";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '.');
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << '%';

    if (current >= total) {
        std::cout << '\n';
    }
    std::cout.flush();
}

float l2_sq(const float* a, const float* b, std::int64_t dim) {
    float sum = 0.0f;
    for (std::int64_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

std::int64_t nearest_centroid(const float* x, const std::vector<float>& centroids, std::int64_t k, std::int64_t dim) {
    std::int64_t best = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (std::int64_t c = 0; c < k; ++c) {
        const float dist = l2_sq(x, centroids.data() + c * dim, dim);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    return best;
}

std::vector<float> kmeans(
    const float* data,
    std::int64_t n,
    std::int64_t dim,
    std::int64_t k,
    std::int64_t iters,
    const char* progress_name = nullptr
) {
    k = std::min(k, n);
    std::vector<float> centroids(static_cast<std::size_t>(k * dim), 0.0f);
    std::vector<std::int64_t> assign(static_cast<std::size_t>(n), 0);
    std::vector<std::int64_t> counts(static_cast<std::size_t>(k), 0);
    std::vector<float> sums(static_cast<std::size_t>(k * dim), 0.0f);

    for (std::int64_t c = 0; c < k; ++c) {
        std::copy(data + c * dim, data + (c + 1) * dim, centroids.data() + c * dim);
    }

    for (std::int64_t iter = 0; iter < iters; ++iter) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0f);

        for (std::int64_t i = 0; i < n; ++i) {
            const std::int64_t c = nearest_centroid(data + i * dim, centroids, k, dim);
            assign[static_cast<std::size_t>(i)] = c;
            ++counts[static_cast<std::size_t>(c)];

            for (std::int64_t j = 0; j < dim; ++j) {
                sums[static_cast<std::size_t>(c * dim + j)] += data[i * dim + j];
            }

            if (progress_name != nullptr && (i % 10000 == 0 || i + 1 == n)) {
                print_kmeans_iter_progress(progress_name, iter + 1, iters, i + 1, n);
            }
        }

        for (std::int64_t c = 0; c < k; ++c) {
            if (counts[static_cast<std::size_t>(c)] == 0) {
                continue;
            }

            for (std::int64_t j = 0; j < dim; ++j) {
                centroids[static_cast<std::size_t>(c * dim + j)] =
                    sums[static_cast<std::size_t>(c * dim + j)] /
                    static_cast<float>(counts[static_cast<std::size_t>(c)]);
            }
        }
    }

    return centroids;
}

std::vector<float> extract_subspace(
    const Dataset& data,
    const std::vector<std::int64_t>& ids,
    const std::vector<std::int64_t>& dims
) {
    const std::int64_t rows = static_cast<std::int64_t>(ids.size());
    const std::int64_t subdim = static_cast<std::int64_t>(dims.size());
    std::vector<float> out(static_cast<std::size_t>(rows * subdim));

    for (std::int64_t r = 0; r < rows; ++r) {
        const float* x = data.data_ptr + ids[static_cast<std::size_t>(r)] * data.d;
        for (std::int64_t j = 0; j < subdim; ++j) {
            out[static_cast<std::size_t>(r * subdim + j)] = x[dims[static_cast<std::size_t>(j)]];
        }
    }

    return out;
}

std::int64_t ceil_log2(std::int64_t value) {
    if (value <= 1) {
        return 0;
    }

    std::int64_t bits = 0;
    std::int64_t x = value - 1;
    while (x > 0) {
        x >>= 1;
        ++bits;
    }
    return bits;
}

float global_variance_sum(const Dataset& data) {
    double total = 0.0;

    for (std::int64_t dim = 0; dim < data.d; ++dim) {
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
        total += var / static_cast<double>(data.n);
    }

    return static_cast<float>(total);
}

std::filesystem::path ivf_cache_path(std::int64_t n, std::int64_t d, std::int64_t ivf_list_size) {
    return std::filesystem::path(kIVFCacheDir) /
        ("ivf_n" + std::to_string(n) +
         "_d" + std::to_string(d) +
         "_lists" + std::to_string(ivf_list_size) +
         ".bin");
}

bool load_ivf_cache(
    const std::filesystem::path& path,
    std::int64_t n,
    std::int64_t d,
    std::int64_t coarse_k,
    std::vector<float>& coarse_centroids,
    std::vector<Index::IVFList>& lists
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[8] = {};
    std::int64_t file_n = 0;
    std::int64_t file_d = 0;
    std::int64_t file_k = 0;

    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&file_n), sizeof(file_n));
    in.read(reinterpret_cast<char*>(&file_d), sizeof(file_d));
    in.read(reinterpret_cast<char*>(&file_k), sizeof(file_k));

    if (!in || std::string(magic, sizeof(magic)) != std::string("APQIVF1", 8) ||
        file_n != n || file_d != d || file_k != coarse_k) {
        return false;
    }

    coarse_centroids.resize(static_cast<std::size_t>(coarse_k * d));
    in.read(
        reinterpret_cast<char*>(coarse_centroids.data()),
        static_cast<std::streamsize>(coarse_centroids.size() * sizeof(float))
    );

    lists.clear();
    lists.resize(static_cast<std::size_t>(coarse_k));
    for (std::int64_t list_id = 0; list_id < coarse_k; ++list_id) {
        std::int64_t list_n = 0;
        in.read(reinterpret_cast<char*>(&list_n), sizeof(list_n));
        auto& ids = lists[static_cast<std::size_t>(list_id)].ids;
        ids.resize(static_cast<std::size_t>(list_n));
        in.read(
            reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(std::int64_t))
        );
    }

    return static_cast<bool>(in);
}

void save_ivf_cache(
    const std::filesystem::path& path,
    std::int64_t n,
    std::int64_t d,
    std::int64_t coarse_k,
    const std::vector<float>& coarse_centroids,
    const std::vector<Index::IVFList>& lists
) {
    std::filesystem::create_directories(path.parent_path());

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return;
    }

    const char magic[8] = {'A', 'P', 'Q', 'I', 'V', 'F', '1', '\0'};
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(&coarse_k), sizeof(coarse_k));
    out.write(
        reinterpret_cast<const char*>(coarse_centroids.data()),
        static_cast<std::streamsize>(coarse_centroids.size() * sizeof(float))
    );

    for (const auto& list : lists) {
        const std::int64_t list_n = static_cast<std::int64_t>(list.ids.size());
        out.write(reinterpret_cast<const char*>(&list_n), sizeof(list_n));
        out.write(
            reinterpret_cast<const char*>(list.ids.data()),
            static_cast<std::streamsize>(list.ids.size() * sizeof(std::int64_t))
        );
    }
}

} // namespace

Index::Index(std::vector<PQSubspaceConfig> config, std::int64_t ivf_list_size)
    : config_(std::move(config)),
      ivf_list_size_(ivf_list_size) {}

Index::Index(
    std::int64_t adaptive_subspace_count,
    std::int64_t ivf_list_size,
    float radius_scale
)
    : ivf_list_size_(ivf_list_size),
      adaptive_subspace_count_(adaptive_subspace_count),
      adaptive_radius_scale_(radius_scale),
      adaptive_config_(true) {}

void Index::validate_config(std::int64_t dim) const {
    if (ivf_list_size_ <= 0) {
        throw std::runtime_error("ivf_list_size must be positive");
    }

    std::vector<unsigned char> seen(static_cast<std::size_t>(dim), 0);
    std::int64_t total_dims = 0;

    for (const auto& sub : config_) {
        if (sub.dims.empty()) {
            throw std::runtime_error("PQ subspace dims cannot be empty");
        }
        if (sub.codebook_size <= 0) {
            throw std::runtime_error("PQ codebook_size must be positive");
        }

        for (const std::int64_t d : sub.dims) {
            if (d < 0 || d >= dim) {
                throw std::runtime_error("PQ dim is out of range");
            }
            if (seen[static_cast<std::size_t>(d)]) {
                throw std::runtime_error("PQ dims contain duplicate dimension");
            }
            seen[static_cast<std::size_t>(d)] = 1;
            ++total_dims;
        }
    }

    if (total_dims != dim) {
        throw std::runtime_error("PQ dims must cover all dataset dimensions exactly once");
    }
}

void Index::build(const Dataset& data) {
    data_ = &data;
    d_ = data.d;
    average_bits_ = 0.0;
    if (!adaptive_config_) {
        validate_config(d_);
    }
    quantization_errors_.assign(static_cast<std::size_t>(data.n), 0.0f);

    const std::int64_t coarse_k = std::min(ivf_list_size_, data.n);
    const std::filesystem::path cache_path = ivf_cache_path(data.n, data.d, coarse_k);

    if (load_ivf_cache(cache_path, data.n, data.d, coarse_k, coarse_centroids_, lists_)) {
        std::cout << "loading IVF cache " << cache_path << "\n";
    } else {
        std::cout << "building IVF coarse kmeans\n";
        coarse_centroids_ = kmeans(data.data_ptr, data.n, data.d, coarse_k, 10, "ivf km ");
        lists_.clear();
        lists_.resize(static_cast<std::size_t>(coarse_k));

        std::cout << "building IVF assignment\n";
        for (std::int64_t i = 0; i < data.n; ++i) {
            const std::int64_t list_id = nearest_centroid(
                data.data_ptr + i * data.d,
                coarse_centroids_,
                coarse_k,
                data.d
            );
            lists_[static_cast<std::size_t>(list_id)].ids.push_back(i);

            if (i % 10000 == 0 || i + 1 == data.n) {
                print_progress("ivf as ", i + 1, data.n);
            }
        }

        std::cout << "saving IVF cache " << cache_path << "\n";
        save_ivf_cache(cache_path, data.n, data.d, coarse_k, coarse_centroids_, lists_);
    }

    std::cout << "building per-list PQ\n";
    double total_bits = 0.0;
    std::int64_t total_encoded = 0;

    // Compute variance for adaptive radius (use residual variance if in residual mode)
    float adaptive_radius_sq = 0.0f;
    if (adaptive_config_) {
        if (residual_ && coarse_k > 0) {
            double sum_sq = 0.0;
            std::int64_t sample_n = std::min(data.n, std::int64_t{10000});
            for (std::int64_t i = 0; i < sample_n; ++i) {
                const std::int64_t c = nearest_centroid(
                    data.data_ptr + i * data.d, coarse_centroids_, coarse_k, data.d);
                for (std::int64_t j = 0; j < data.d; ++j) {
                    const float diff = data.data_ptr[i * data.d + j] -
                                       coarse_centroids_[static_cast<std::size_t>(c * data.d + j)];
                    sum_sq += static_cast<double>(diff * diff);
                }
            }
            adaptive_radius_sq = adaptive_radius_scale_ *
                static_cast<float>(sum_sq / static_cast<double>(sample_n));
        } else {
            adaptive_radius_sq = adaptive_radius_scale_ * global_variance_sum(data);
        }
        std::cout << "adaptive radius_sq=" << adaptive_radius_sq << "\n";
    }

    for (std::int64_t list_id = 0; list_id < static_cast<std::int64_t>(lists_.size()); ++list_id) {
        auto& list = lists_[static_cast<std::size_t>(list_id)];
        const std::int64_t list_n = static_cast<std::int64_t>(list.ids.size());

        if (list_n == 0) {
            print_list_progress(list_id + 1, static_cast<std::int64_t>(lists_.size()));
            continue;
        }

        // Build residual dataset if in residual mode
        std::unique_ptr<Dataset> residual_dataset;
        std::vector<std::int64_t> seq_ids;
        if (residual_) {
            residual_dataset = std::make_unique<Dataset>(list_n, d_);
            for (std::int64_t row = 0; row < list_n; ++row) {
                const std::int64_t orig_id = list.ids[static_cast<std::size_t>(row)];
                for (std::int64_t j = 0; j < d_; ++j) {
                    residual_dataset->data_ptr[static_cast<std::size_t>(row * d_ + j)] =
                        data.data_ptr[orig_id * d_ + j] -
                        coarse_centroids_[static_cast<std::size_t>(list_id * d_ + j)];
                }
            }
            seq_ids.resize(static_cast<std::size_t>(list_n));
            std::iota(seq_ids.begin(), seq_ids.end(), 0);
        }

        const Dataset& train_data = residual_ ? *residual_dataset : data;
        const std::vector<std::int64_t>& train_ids = residual_ ? seq_ids : list.ids;

        if (adaptive_config_) {
            list.config = make_kcenter_merge_config(
                train_data,
                train_ids,
                adaptive_subspace_count_,
                adaptive_radius_sq
            );
        } else {
            list.config = config_;
        }

        const std::int64_t subspace_count = static_cast<std::int64_t>(list.config.size());
        list.quantizers.resize(static_cast<std::size_t>(subspace_count));
        list.codes.assign(static_cast<std::size_t>(list_n * subspace_count), 0);

        for (std::int64_t s = 0; s < subspace_count; ++s) {
            const auto& sub = list.config[static_cast<std::size_t>(s)];
            auto& q = list.quantizers[static_cast<std::size_t>(s)];
            q.dim = static_cast<std::int64_t>(sub.dims.size());
            q.k = std::min(sub.codebook_size, list_n);
            q.k = std::min<std::int64_t>(q.k, std::numeric_limits<std::int16_t>::max());

            const std::vector<float> sub_data = extract_subspace(train_data, train_ids, sub.dims);
            q.centroids = kmeans(sub_data.data(), list_n, q.dim, q.k, 10);

            for (std::int64_t row = 0; row < list_n; ++row) {
                const std::int64_t code = nearest_centroid(
                    sub_data.data() + row * q.dim,
                    q.centroids,
                    q.k,
                    q.dim
                );
                list.codes[static_cast<std::size_t>(row * subspace_count + s)] =
                    static_cast<std::int16_t>(code);
                quantization_errors_[static_cast<std::size_t>(list.ids[static_cast<std::size_t>(row)])] +=
                    l2_sq(
                        sub_data.data() + row * q.dim,
                        q.centroids.data() + code * q.dim,
                        q.dim
                    );
            }
        }

        std::int64_t bits_per_vector = 0;
        for (const auto& sub : list.config) {
            bits_per_vector += ceil_log2(sub.codebook_size);
        }
        total_bits += static_cast<double>(bits_per_vector) * static_cast<double>(list_n);
        total_encoded += list_n;

        print_list_progress(list_id + 1, static_cast<std::int64_t>(lists_.size()));
    }

    if (total_encoded > 0) {
        average_bits_ = total_bits / static_cast<double>(total_encoded);
    }
}

const std::vector<float>& Index::quantization_errors() const {
    return quantization_errors_;
}

double Index::average_bits() const {
    return average_bits_;
}

void Index::export_list_configs(const std::string& path, std::int64_t max_lists) const {
    std::ofstream out(path);
    if (!out) {
        return;
    }

    out << "{\n";
    out << "  \"max_lists\": " << max_lists << ",\n";
    out << "  \"lists\": [\n";

    std::int64_t written = 0;
    for (std::int64_t list_id = 0;
         list_id < static_cast<std::int64_t>(lists_.size()) && written < max_lists;
         ++list_id) {
        const auto& list = lists_[static_cast<std::size_t>(list_id)];
        if (list.ids.empty()) {
            continue;
        }

        if (written > 0) {
            out << ",\n";
        }

        out << "    {\n";
        out << "      \"list_id\": " << list_id << ",\n";
        out << "      \"size\": " << list.ids.size() << ",\n";
        out << "      \"subspaces\": [\n";

        for (std::int64_t s = 0; s < static_cast<std::int64_t>(list.config.size()); ++s) {
            const auto& sub = list.config[static_cast<std::size_t>(s)];
            const auto& q = list.quantizers[static_cast<std::size_t>(s)];

            out << "        {\"id\": " << s
                << ", \"codebook_size\": " << sub.codebook_size
                << ", \"actual_k\": " << q.k
                << ", \"dims\": [";

            for (std::int64_t i = 0; i < static_cast<std::int64_t>(sub.dims.size()); ++i) {
                if (i > 0) {
                    out << ", ";
                }
                out << sub.dims[static_cast<std::size_t>(i)];
            }

            out << "]}";
            if (s + 1 < static_cast<std::int64_t>(list.config.size())) {
                out << ",";
            }
            out << "\n";
        }

        out << "      ]\n";
        out << "    }";
        ++written;
    }

    out << "\n";
    out << "  ]\n";
    out << "}\n";
}

std::vector<SearchResult> Index::query(const std::vector<float>& query_vector, std::int64_t nprobe) const {
    return query(query_vector.data(), nprobe);
}

std::vector<SearchResult> Index::query(const float* query_vector, std::int64_t nprobe) const {
    const std::int64_t coarse_k = static_cast<std::int64_t>(lists_.size());
    nprobe = std::min(nprobe, coarse_k);

    std::vector<SearchResult> coarse(static_cast<std::size_t>(coarse_k));
    for (std::int64_t c = 0; c < coarse_k; ++c) {
        coarse[static_cast<std::size_t>(c)] = {
            c,
            l2_sq(query_vector, coarse_centroids_.data() + c * d_, d_)
        };
    }

    std::partial_sort(
        coarse.begin(),
        coarse.begin() + nprobe,
        coarse.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    std::vector<SearchResult> results;
    std::vector<std::vector<float>> tables;
    std::vector<std::vector<float>> query_subspaces;

    for (std::int64_t probe = 0; probe < nprobe; ++probe) {
        const auto& list = lists_[static_cast<std::size_t>(coarse[static_cast<std::size_t>(probe)].id)];
        const std::int64_t list_n = static_cast<std::int64_t>(list.ids.size());
        if (list_n == 0) {
            continue;
        }

        const std::int64_t subspace_count = static_cast<std::int64_t>(list.config.size());
        tables.resize(static_cast<std::size_t>(subspace_count));
        query_subspaces.resize(static_cast<std::size_t>(subspace_count));

        for (std::int64_t s = 0; s < subspace_count; ++s) {
            const auto& sub = list.config[static_cast<std::size_t>(s)];
            const auto& q = list.quantizers[static_cast<std::size_t>(s)];
            auto& query_sub = query_subspaces[static_cast<std::size_t>(s)];
            auto& table = tables[static_cast<std::size_t>(s)];
            query_sub.resize(static_cast<std::size_t>(q.dim));
            table.resize(static_cast<std::size_t>(q.k));

            for (std::int64_t j = 0; j < q.dim; ++j) {
                const std::int64_t dim_idx = sub.dims[static_cast<std::size_t>(j)];
                float val = query_vector[dim_idx];
                if (residual_) {
                    val -= coarse_centroids_[
                        static_cast<std::size_t>(coarse[static_cast<std::size_t>(probe)].id * d_ + dim_idx)];
                }
                query_sub[static_cast<std::size_t>(j)] = val;
            }

            for (std::int64_t c = 0; c < q.k; ++c) {
                float dist = 0.0f;
                const float* centroid = q.centroids.data() + c * q.dim;
                for (std::int64_t j = 0; j < q.dim; ++j) {
                    const float diff = query_sub[static_cast<std::size_t>(j)] - centroid[j];
                    dist += diff * diff;
                }
                table[static_cast<std::size_t>(c)] = dist;
            }
        }

        for (std::int64_t row = 0; row < list_n; ++row) {
            float dist = 0.0f;
            for (std::int64_t s = 0; s < subspace_count; ++s) {
                const std::int16_t code = list.codes[static_cast<std::size_t>(row * subspace_count + s)];
                dist += tables[static_cast<std::size_t>(s)][static_cast<std::size_t>(code)];
            }
            results.push_back({list.ids[static_cast<std::size_t>(row)], dist});
        }
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });

    return results;
}

QueryResult Index::query_refine(
    const float* query_vector,
    std::int64_t nprobe,
    std::int64_t topk,
    float alpha
) const {
    const std::int64_t coarse_k = static_cast<std::int64_t>(lists_.size());
    nprobe = std::min(nprobe, coarse_k);

    std::vector<SearchResult> coarse(static_cast<std::size_t>(coarse_k));
    for (std::int64_t c = 0; c < coarse_k; ++c) {
        coarse[static_cast<std::size_t>(c)] = {
            c,
            l2_sq(query_vector, coarse_centroids_.data() + c * d_, d_)
        };
    }

    std::partial_sort(
        coarse.begin(),
        coarse.begin() + nprobe,
        coarse.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    QueryResult out;
    std::vector<SearchResult> heap;
    heap.reserve(static_cast<std::size_t>(topk));

    auto worse_first = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    };

    auto push_exact = [&](std::int64_t id, float exact_dist) {
        SearchResult item{id, exact_dist};
        if (static_cast<std::int64_t>(heap.size()) < topk) {
            heap.push_back(item);
            std::push_heap(heap.begin(), heap.end(), worse_first);
            return;
        }

        if (exact_dist < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end(), worse_first);
            heap.back() = item;
            std::push_heap(heap.begin(), heap.end(), worse_first);
        }
    };

    std::vector<std::vector<float>> tables;
    std::vector<std::vector<float>> query_subspaces;

    for (std::int64_t probe = 0; probe < nprobe; ++probe) {
        const auto& list = lists_[static_cast<std::size_t>(coarse[static_cast<std::size_t>(probe)].id)];
        const std::int64_t list_n = static_cast<std::int64_t>(list.ids.size());
        if (list_n == 0) {
            continue;
        }

        const std::int64_t subspace_count = static_cast<std::int64_t>(list.config.size());
        tables.resize(static_cast<std::size_t>(subspace_count));
        query_subspaces.resize(static_cast<std::size_t>(subspace_count));

        for (std::int64_t s = 0; s < subspace_count; ++s) {
            const auto& sub = list.config[static_cast<std::size_t>(s)];
            const auto& q = list.quantizers[static_cast<std::size_t>(s)];
            auto& query_sub = query_subspaces[static_cast<std::size_t>(s)];
            auto& table = tables[static_cast<std::size_t>(s)];
            query_sub.resize(static_cast<std::size_t>(q.dim));
            table.resize(static_cast<std::size_t>(q.k));

            for (std::int64_t j = 0; j < q.dim; ++j) {
                const std::int64_t dim_idx = sub.dims[static_cast<std::size_t>(j)];
                float val = query_vector[dim_idx];
                if (residual_) {
                    val -= coarse_centroids_[
                        static_cast<std::size_t>(coarse[static_cast<std::size_t>(probe)].id * d_ + dim_idx)];
                }
                query_sub[static_cast<std::size_t>(j)] = val;
            }

            for (std::int64_t c = 0; c < q.k; ++c) {
                float dist = 0.0f;
                const float* centroid = q.centroids.data() + c * q.dim;
                for (std::int64_t j = 0; j < q.dim; ++j) {
                    const float diff = query_sub[static_cast<std::size_t>(j)] - centroid[j];
                    dist += diff * diff;
                }
                table[static_cast<std::size_t>(c)] = dist;
            }
        }

        for (std::int64_t row = 0; row < list_n; ++row) {
            ++out.scanned_count;
            float pq_dist = 0.0f;
            for (std::int64_t s = 0; s < subspace_count; ++s) {
                const std::int16_t code = list.codes[static_cast<std::size_t>(row * subspace_count + s)];
                pq_dist += tables[static_cast<std::size_t>(s)][static_cast<std::size_t>(code)];
            }

            const std::int64_t id = list.ids[static_cast<std::size_t>(row)];
            if (static_cast<std::int64_t>(heap.size()) >= topk) {
                const float pq_l2 = std::sqrt(pq_dist);
                const float qe_l2 = std::sqrt(quantization_errors_[static_cast<std::size_t>(id)]);
                const float bound_l2 = pq_l2 - alpha * qe_l2;
                if (bound_l2 * bound_l2 > heap.front().distance) {
                    ++out.skipped_count;
                    continue;
                }
            }

            ++out.exact_count;
            const float exact_dist = l2_sq(
                query_vector,
                data_->data_ptr + id * data_->d,
                data_->d
            );
            push_exact(id, exact_dist);
        }
    }

    std::sort_heap(heap.begin(), heap.end(), worse_first);
    out.results = std::move(heap);
    return out;
}

