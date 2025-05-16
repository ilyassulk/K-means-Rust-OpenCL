#!/usr/bin/env python3
"""
Замер времени CPU vs GPU для нашего gpu_kmeans.

Пример:
$ python benchmark_kmeans.py --build
"""
import subprocess, os, csv, sys, time, re, itertools
from typing import List, Dict, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN   = os.path.join(ROOT, "target", "release", "gpu_kmeans")
GEN   = os.path.join(ROOT, "generate_data_kmeans.py")
RESULT_FILE = os.path.join(ROOT, "results_kmeans.csv")
PY = sys.executable or "python3"

# Параметры бенча
N_VALUES = [10_000, 50_000, 100_000]
D_VALUES = [2, 8, 32]
K_VALUES = [4, 8, 16]
ITERATIONS = 50
EPS        = 1e-3


def gen_data(n: int, d: int, k: int) -> str:
    dst = os.path.join(ROOT, f"data_{n}_{d}_{k}.csv")
    if not os.path.exists(dst):
        subprocess.run([PY, GEN, "-n", str(n), "-f", str(d), "-k", str(k), "-o", dst],
                       check=True)
    return dst


def run(binary: str, data: str, k: int, gpu: bool) -> float:
    cmd = [
        binary,
        "-f", data,
        "-k", str(k),
        "-i", str(ITERATIONS),
        "-e", str(EPS),
    ]
    if gpu:
        cmd.append("-o")

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print("❌", res.stderr); return float("inf")

    # ищем строку «Total GPU time: 0.123s» или «CPU done in …»
    m = re.search(r"([0-9.]+)\s*s", res.stdout)
    return float(m.group(1)) if m else elapsed


def bench() -> List[Dict[str, float]]:
    rows = []
    cases = list(itertools.product(N_VALUES, D_VALUES, K_VALUES))
    for n, d, k in cases:
        data = gen_data(n, d, k)
        print(f"\n··· N={n}, D={d}, K={k}")
        t_cpu = run(BIN, data, k, gpu=False)
        print(f"   CPU  {t_cpu:.3f} s")
        t_gpu = run(BIN, data, k, gpu=True)
        print(f"   GPU  {t_gpu:.3f} s  (×{t_cpu/t_gpu:,.1f})")
        rows.append(dict(N=n, D=d, K=k,
                         Iter=ITERATIONS, EPS=EPS,
                         CPU=t_cpu, GPU=t_gpu, Speedup=t_cpu/t_gpu))
    return rows


def save(rows: List[Dict]):
    fresh = not os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if fresh: w.writeheader()
        w.writerows(rows)
    print("↳ результаты в", RESULT_FILE)


if __name__ == "__main__":
    # при первом запуске полезно добавить ключ --build
    if "--build" in sys.argv:
        print("· Компиляция gpu_kmeans …")
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)

    out = bench()
    save(out)
