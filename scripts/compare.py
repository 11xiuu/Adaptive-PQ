#!/usr/bin/env python3

import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(file_path):
    """加载 JSON 结果文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "result"
    
    # 1. 定义输入文件路径
    # 修改点：使用包含 Adaptive 结果的 benchmark_results.json
    cpp_path = result_dir / "benchmark_results.json"
    py_path = result_dir / "faiss_standard_ivfpq_sift1m.json"
    
    # 2. 输出图片路径
    out_path = result_dir / "compare_cpp_adaptive_vs_faiss.png"

    # 3. 加载数据
    print(f"Loading C++ results from: {cpp_path.name}")
    data_cpp = load_results(cpp_path)
    
    print(f"Loading Python (Faiss) results from: {py_path.name}")
    data_py = load_results(py_path)

    if not data_cpp or not data_py:
        print("❌ Error: Could not load one or both JSON files. Please check paths.")
        return

    # 4. 开始绘图
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # 修改点：扩展颜色映射，覆盖所有 C++ 配置
    # 8x8bit: 蓝色, 16x4bit: 橙色, kcenter8: 绿色, kcenter16: 红色
    colors = {
        '8x8bit': '#1f77b4', 
        '16x4bit': '#ff7f0e',
        'kcenter8': '#2ca02c',
        'kcenter16': '#d62728'
    } 
    
    markers_cpp = 'o'
    markers_py = 'x'

    def plot_data(data, source_label, marker_style, is_faiss=False):
        for cfg in data["configs"]:
            cfg_name = cfg['name']
            color = colors.get(cfg_name, '#333333') # 默认灰色
            
            # 获取该配置下所有的 alpha
            # Faiss 数据中可能没有 alpha 字段，或者默认为 0.6，这里做兼容处理
            if is_faiss:
                alphas = [0.6] 
            else:
                alphas = sorted({row.get("alpha", 0.6) for row in cfg["results"]}, reverse=True)
            
            for alpha in alphas:
                rows = [row for row in cfg["results"] if row.get("alpha", 0.6) == alpha]
                if not rows:
                    continue
                    
                rows.sort(key=lambda row: row["nprobe"])
                
                time_ms = [row["time_ms"] for row in rows]
                recall = [row["recall"] for row in rows]
                
                # 构造标签
                # 如果是 C++ 且 alpha != 0.6，显示 alpha 值以便区分
                if not is_faiss and alpha != 0.6:
                    label = f"{cfg_name} α={alpha} ({source_label})"
                else:
                    label = f"{cfg_name} ({source_label})"
                
                # 绘图
                ax.plot(time_ms, recall, 
                        marker=marker_style, 
                        linestyle='-', 
                        color=color,
                        label=label, 
                        markersize=6,
                        linewidth=1.5)

    # 绘制 C++ 数据 (实线圆点)
    plot_data(data_cpp, "C++", markers_cpp, is_faiss=False)
    
    # 绘制 Python/Faiss 数据 (虚线叉号)
    plot_data(data_py, "Faiss", markers_py, is_faiss=True)

    # 5. 设置图表样式
    topk = data_cpp.get('topk', 10)
    ax.set_title(f"Performance Comparison: Adaptive PQ (C++) vs Standard IVF-PQ (Faiss)", fontsize=14, pad=20)
    ax.set_xlabel("Average Time per Query (ms)", fontsize=12)
    ax.set_ylabel(f"Recall@{topk}", fontsize=12)
    
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # 图例放在右下角外部或内部，防止遮挡
    ax.legend(loc="lower right", fontsize=9, title="Config & Source", ncol=2)
    
    # 调整布局
    fig.tight_layout()
    
    # 6. 保存并显示
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {out_path}")
    
    # 本地运行查看图表
    plt.show()

if __name__ == "__main__":
    main()