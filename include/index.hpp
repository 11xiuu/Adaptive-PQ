#pragma once

#include "dataset.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct PQSubspaceConfig {
    std::vector<std::int64_t> dims;
    std::int64_t codebook_size = 0;
};

struct SearchResult {
    std::int64_t id = -1;
    float distance = 0.0f;
};

struct QueryResult {
    std::vector<SearchResult> results;
    std::int64_t scanned_count = 0;
    std::int64_t exact_count = 0;
    std::int64_t skipped_count = 0;
};

class Index {
public:
    Index(std::vector<PQSubspaceConfig> config, std::int64_t ivf_list_size);
    Index(
        std::int64_t adaptive_subspace_count,
        std::int64_t ivf_list_size,
        float radius_scale
    );

    void build(const Dataset& data);

    std::vector<SearchResult> query(const float* query_vector, std::int64_t nprobe) const;
    std::vector<SearchResult> query(const std::vector<float>& query_vector, std::int64_t nprobe) const;
    QueryResult query_refine(
        const float* query_vector,
        std::int64_t nprobe,
        std::int64_t topk,
        float alpha
    ) const;
    const std::vector<float>& quantization_errors() const;
    double average_bits() const;
    void export_list_configs(const std::string& path, std::int64_t max_lists) const;

public:
    struct SubQuantizer {
        std::int64_t dim = 0;
        std::int64_t k = 0;
        std::vector<float> centroids;
    };

    struct IVFList {
        std::vector<std::int64_t> ids;
        std::vector<PQSubspaceConfig> config;
        std::vector<SubQuantizer> quantizers;
        std::vector<std::int16_t> codes;
    };

private:
    std::vector<PQSubspaceConfig> config_;
    std::int64_t ivf_list_size_ = 0;
    std::int64_t adaptive_subspace_count_ = 0;
    float adaptive_radius_scale_ = 0.5f;
    bool adaptive_config_ = false;
    std::int64_t d_ = 0;
    const Dataset* data_ = nullptr;
    std::vector<float> coarse_centroids_;
    std::vector<IVFList> lists_;
    std::vector<float> quantization_errors_;
    double average_bits_ = 0.0;

    void validate_config(std::int64_t dim) const;
};
