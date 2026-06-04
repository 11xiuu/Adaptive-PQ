#pragma once

#include <cstdint>
#include <string>

class Dataset {
public:
    std::int64_t n = 0;
    std::int64_t d = 0;
    float* data_ptr = nullptr;

    Dataset() = default;
    Dataset(std::int64_t rows, std::int64_t dim);
    explicit Dataset(const std::string& path, std::int64_t max_n = -1);

    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    Dataset(Dataset&& other) noexcept;
    Dataset& operator=(Dataset&& other) noexcept;

    ~Dataset();

    static Dataset readVectorFromDirectory(const std::string& path, std::int64_t max_n = -1);

        // 添加这个静态方法声明
    static Dataset readFvecs(const std::string& path, std::int64_t max_n = -1);
};
