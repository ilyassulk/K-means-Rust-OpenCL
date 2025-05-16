#!/usr/bin/env python3
"""
Генератор входных данных для GPU-k-means

* Cохраняет только признаки (без столбца label), потому что k-means
  не требует известных меток.
* Для 2-D визуализирует исходные точки и случайно выбранные
  начальные центроиды (первые K точек) — удобно для отчёта.
"""
import random, os, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_points(n: int, d: int, k: int, out_path: str) -> None:
    X, y = make_blobs(
        n_samples=n,
        n_features=d,
        centers=k,
        cluster_std=1.0,
        random_state=random.randrange(1, 1000),
    )

    # Сохраняем ТОЛЬКО признаки
    pd.DataFrame(X).to_csv(out_path, index=False, header=False)
    print(f"Точки сохранены в: {out_path}")

    # Для контроля — сохраняем «истинные» метки отдельно (необязательно)
    truth_path = out_path.replace(".csv", "_labels.csv")
    pd.DataFrame(y).to_csv(truth_path, index=False, header=False)
    print(f"Истинные метки (для анализа) → {truth_path}")

    # График (только в 2-D)
    if d == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7)
        # первые K точек — именно их берёт наше Rust-приложение
        plt.scatter(
            X[:k, 0], X[:k, 1],
            c="red", marker="X", s=140, label="initial centroids"
        )
        plt.title(f"Синтетические данные для k-means (K={k})")
        plt.legend(); plt.grid(True)
        img_path = out_path.replace(".csv", ".png")
        plt.savefig(img_path)
        print(f"Диаграмма → {img_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("generate_data_kmeans")
    ap.add_argument("-n", "--num_points", type=int, default=50_000)
    ap.add_argument("-f", "--num_features", type=int, default=2)
    ap.add_argument("-k", "--num_clusters", type=int, default=8)
    ap.add_argument("-o", "--output", default="points.csv")
    args = ap.parse_args()

    assert args.num_points > 0 and args.num_features > 0 and args.num_clusters > 0
    generate_points(args.num_points, args.num_features, args.num_clusters, args.output)
