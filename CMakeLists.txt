cmake_minimum_required(VERSION 3.16)

project(AdaptivePQ LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(FAISS_ROOT "/home/zhaixiue/codes/faiss")
set(FAISS_INCLUDE_DIR ${FAISS_ROOT})
set(FAISS_LIB_DIR ${FAISS_ROOT}/build/faiss)

include_directories(${FAISS_INCLUDE_DIR})

link_directories(${FAISS_LIB_DIR})

add_library(dataset
    src/config_generator.cpp
    src/dataset.cpp
    src/index.cpp
    src/kcenter.cpp
    src/kcenter_benchmark.cpp
)

target_include_directories(dataset
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(dataset PUBLIC cxx_std_17)

target_compile_options(dataset PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-O3 -march=native>
)

add_executable(adaptivepq
    main.cpp
)

target_link_libraries(adaptivepq PRIVATE dataset faiss)

target_compile_options(adaptivepq PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-O3 -march=native>
)

option(ADAPTIVEPQ_BUILD_KCENTER_TEST "Build test_kcenter.cpp when pq.hpp is available" OFF)

if(ADAPTIVEPQ_BUILD_KCENTER_TEST)
    add_executable(test_kcenter test_kcenter.cpp)
    target_link_libraries(test_kcenter PRIVATE dataset)
endif()
