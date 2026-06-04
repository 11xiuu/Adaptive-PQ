#include "dataset.hpp"
#include <cstring> 
#include <algorithm>
#include <fstream>

Dataset::Dataset(std::int64_t rows, std::int64_t dim)
    : n(rows), d(dim), data_ptr(new float[static_cast<std::size_t>(rows * dim)]) {}

Dataset::Dataset(const std::string& path, std::int64_t max_n)
    : Dataset(readVectorFromDirectory(path, max_n)) {}

Dataset::Dataset(Dataset&& other) noexcept
    : n(other.n), d(other.d), data_ptr(other.data_ptr) {
    other.n = 0;
    other.d = 0;
    other.data_ptr = nullptr;
}

Dataset& Dataset::operator=(Dataset&& other) noexcept {
    if (this != &other) {
        delete[] data_ptr;
        n = other.n;
        d = other.d;
        data_ptr = other.data_ptr;
        other.n = 0;
        other.d = 0;
        other.data_ptr = nullptr;
    }
    return *this;
}

Dataset::~Dataset() {
    delete[] data_ptr;
}

Dataset Dataset::readVectorFromDirectory(const std::string& path, std::int64_t max_n) {
    std::ifstream file(path, std::ios::binary);

    std::int64_t n = 0;
    std::int64_t dim = 0;
    file.read(reinterpret_cast<char*>(&n), sizeof(std::int64_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(std::int64_t));

    if (max_n != -1 && n > max_n) {
        n = max_n;
    }

    Dataset result(n, dim);
    float* data = result.data_ptr;

    const std::int64_t total_bytes = n * dim * static_cast<std::int64_t>(sizeof(float));
    constexpr std::int64_t batch_size = 1024 * 1024 * 32;
    std::int64_t offset = 0;

    while (offset < total_bytes) {
        const std::int64_t remaining = total_bytes - offset;
        const std::int64_t to_read = std::min(batch_size, remaining);
        file.read(reinterpret_cast<char*>(data) + offset, to_read);
        offset += to_read;
    }

    return result;
}

Dataset Dataset::readFvecs(const std::string& path, std::int64_t max_n) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // 1. 先读取第一个向量的维度，确定 d
    std::int32_t dim = 0;
    file.read(reinterpret_cast<char*>(&dim), sizeof(std::int32_t));
    if (file.fail() || dim <= 0) {
        throw std::runtime_error("Invalid .fvecs file format or empty file");
    }
    
    // 回退指针到文件开头，以便重新读取所有数据
    file.seekg(0, std::ios::beg);

    // 2. 计算文件大致大小以确定向量数量 (可选，或者直接遍历)
    // 为了简单和兼容 max_n，我们动态读取或预计算
    // 这里采用预计算总行数的方法（如果文件很大，seekg 到末尾可能较慢，但比逐行解析快）
    auto begin_pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    std::size_t file_size = static_cast<std::size_t>(end_pos - begin_pos);
    file.seekg(0, std::ios::beg);

    // 每个向量占用: 4 bytes (dim) + dim * 4 bytes (floats)
    std::size_t vector_bytes = 4 + dim * 4;
    std::int64_t total_vectors = static_cast<std::int64_t>(file_size / vector_bytes);

    if (max_n != -1 && max_n < total_vectors) {
        total_vectors = max_n;
    }

    Dataset result(total_vectors, dim);
    
    // 3. 逐个向量读取
    for (std::int64_t i = 0; i < total_vectors; ++i) {
        std::int32_t current_dim = 0;
        file.read(reinterpret_cast<char*>(&current_dim), sizeof(std::int32_t));
        
        if (current_dim != dim) {
             // 标准 fvecs 每个向量维度应一致，如果不一致需特殊处理，通常 SIFT 是一致的
             throw std::runtime_error("Dimension mismatch in fvecs file at vector " + std::to_string(i));
        }

        file.read(reinterpret_cast<char*>(result.data_ptr + i * dim), dim * sizeof(float));
        
        if (file.fail()) {
            throw std::runtime_error("Error reading vector data at index " + std::to_string(i));
        }
    }

    return result;
}