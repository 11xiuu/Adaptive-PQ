import faiss
import numpy as np
import time
import os
import json


# ==========================================================
# fvecs
# ==========================================================

def load_fvecs(filename):
    vectors = []

    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)

            if not dim_bytes:
                break

            d = np.frombuffer(
                dim_bytes,
                dtype=np.int32
            )[0]

            vec = np.frombuffer(
                f.read(d * 4),
                dtype=np.float32
            )

            vectors.append(vec)

    return np.vstack(vectors).astype('float32')


# ==========================================================
# ivecs
# ==========================================================

def load_ivecs(filename):

    data = np.fromfile(
        filename,
        dtype=np.int32
    )

    dim = data[0]

    data = data.reshape(
        -1,
        dim + 1
    )

    return data[:, 1:]


# ==========================================================
# Config
# ==========================================================

DATASET_DIR = "D:/project/vscodeProjects/AdaptivePQ/data/sift"

MAX_BASE_N = 1000000

NQUERY = 100

TOPK = 10                # 最终精排Top10
TOPK_CANDIDATE = 1000    # 新增：粗排先取Top100候选

NLIST = 1000

NPROBES = [
    1, 2, 3, 4, 5,
    7, 10, 15, 20,
    30, 40, 50
]

PQ_CONFIGS = [
    {
        "name": "8x8bit",
        "M": 8,
        "nbits": 8
    },
    {
        "name": "16x4bit",
        "M": 16,
        "nbits": 4
    }
]

np.random.seed(1234)

try:
    faiss.omp_set_num_threads(1)
except:
    pass


# ==========================================================
# Load Data
# ==========================================================

print("Loading data...")

xb = load_fvecs(
    os.path.join(
        DATASET_DIR,
        "sift_base.fvecs"
    )
)[:MAX_BASE_N]

xq = load_fvecs(
    os.path.join(
        DATASET_DIR,
        "sift_query.fvecs"
    )
)[:NQUERY]

xlearn = load_fvecs(
    os.path.join(
        DATASET_DIR,
        "sift_learn.fvecs"
    )
)

I_gt = load_ivecs(
    os.path.join(
        DATASET_DIR,
        "sift_groundtruth.ivecs"
    )
)[:NQUERY, :TOPK]

D = xb.shape[1]

print(f"Base    : {xb.shape}")
print(f"Query   : {xq.shape}")
print(f"Learn   : {xlearn.shape}")
print(f"GT      : {I_gt.shape}")
print(f"Dim     : {D}")


# ==========================================================
# Benchmark
# ==========================================================

final_output = {
    "dataset_dir": DATASET_DIR,
    "base_n": int(xb.shape[0]),
    "dimension": int(D),
    "topk": TOPK,
    "nlist": NLIST,
    "configs": []
}

for cfg in PQ_CONFIGS:

    name = cfg["name"]
    M = cfg["M"]
    NBITS = cfg["nbits"]

    print("\n" + "=" * 70)
    print(f"Testing {name}")
    print("=" * 70)

    quantizer = faiss.IndexFlatL2(D)

    index = faiss.IndexIVFPQ(
        quantizer,
        D,
        NLIST,
        M,
        NBITS
    )

    print(
        f"Training on "
        f"{len(xlearn)} learn vectors"
    )

    build_start = time.time()

    # ==================================================
    # Train on official learn set
    # ==================================================

    index.train(xlearn)

    index.add(xb)

    build_ms = (
        time.time() - build_start
    ) * 1000

    print(
        f"IndexIVFPQ("
        f"nlist={NLIST}, "
        f"M={M}, "
        f"nbits={NBITS}, "
        f"code_size={index.code_size} bytes)"
    )

    print(
        f"Build Time: "
        f"{build_ms:.2f} ms"
    )

    print()
    print(
        f"{'nprobe':<8}"
        f"{'Time(ms)':<12}"
        f"{'QPS':<12}"
        f"{'Recall@10':<12}"
    )

    print("-" * 45)

    results = []

    for nprobe in NPROBES:

        index.nprobe = nprobe

        # warmup
        index.search(
            xq[:1],
            TOPK_CANDIDATE  # 预热也用100候选
        )

        runs = 5

        start = time.time()

        for _ in range(runs):
            # ===================== 核心修改1：粗排取 Top100 候选 =====================
            D_candidate, I_candidate = index.search(xq, TOPK_CANDIDATE)
            
            # ===================== 核心修改2：精排 Top10 =====================
            I_res_final = []
            for i in range(NQUERY):
                # 获取当前查询的100个候选ID
                candidate_ids = I_candidate[i]
                # 过滤无效ID
                valid_mask = candidate_ids != -1
                candidate_ids = candidate_ids[valid_mask]
                # 计算精确欧式距离
                exact_dists = np.sum((xq[i] - xb[candidate_ids]) ** 2, axis=1)
                # 精排后取 Top10
                top10_idx = np.argsort(exact_dists)[:TOPK]
                I_res_final.append(candidate_ids[top10_idx])
            I_res = np.array(I_res_final)

        elapsed = (
            time.time() - start
        ) / runs

        avg_time_ms = (
            elapsed * 1000
        ) / NQUERY

        qps = NQUERY / elapsed

        recalls = []

        for i in range(NQUERY):

            hit = np.intersect1d(
                I_res[i],
                I_gt[i]
            )

            recalls.append(
                len(hit) / TOPK
            )

        avg_recall = float(
            np.mean(recalls)
        )

        print(
            f"{nprobe:<8}"
            f"{avg_time_ms:<12.4f}"
            f"{qps:<12.1f}"
            f"{avg_recall:<12.4f}"
        )

        results.append(
            {
                "alpha": 0.6,
                "nprobe": int(nprobe),

                "time_ms":
                    round(avg_time_ms, 5),

                "qps":
                    round(qps, 2),

                "recall":
                    round(avg_recall, 6),

                "average_bits":
                    int(M * NBITS),

                "exact_per_query": 0,
                "skipped_per_query": 0,
                "scanned_per_query": 0,
                "exact_ratio": 0,
                "skipped_ratio": 0
            }
        )

    final_output["configs"].append(
        {
            "name": name,

            "subspaces":
                int(M),

            "average_bits":
                int(M * NBITS),

            "build_ms":
                round(build_ms, 2),

            "results":
                results
        }
    )


# ==========================================================
# Save
# ==========================================================

output_path = (
    "D:/project/vscodeProjects/AdaptivePQ/result/"
    "faiss_standard_ivfpq_sift1m.json"
)

with open(
    output_path,
    "w"
) as f:

    json.dump(
        final_output,
        f,
        indent=2
    )

print("\n")
print("=" * 70)
print("Benchmark Finished")
print("=" * 70)
print(f"Saved to: {output_path}")