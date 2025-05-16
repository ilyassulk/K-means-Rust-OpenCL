//! GPU k‑means (OpenCL 1.2‑friendly):
//! • Assignment‑step выполняется на GPU без атомиков / расширений.
//! • Update‑step (пересчёт центроидов) делается на CPU — это совместимо даже
//!   с драйверами, где нет `atomic_add` / `work_group_reduce_*`.
//! • CSV‑файл содержит **только признаки**, без столбца меток.

use std::{error::Error, fs::File, path::Path, time::Instant};
use clap::Parser;
use csv::ReaderBuilder;
use ocl::{Buffer, Kernel, ProQue};

#[derive(Debug, Clone)]
struct Point {
    features: Vec<f32>,
}

#[derive(Parser, Debug)]
struct Args {
    /// CSV‑файл с признаками
    #[clap(short = 'f', long)]
    file: String,
    /// Кол‑во кластеров K
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Максимум итераций
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости (макс. изменение координаты)
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    /// Использовать GPU‑режим (иначе — чистый CPU)
    #[clap(short = 'o', long)]
    opencl: bool,
}

/// Чтение CSV: каждая строка = вектор признаков `f32`.
fn read_points<P: AsRef<Path>>(path: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(File::open(path)?);
    let mut pts = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let feats: Vec<f32> = rec
            .iter()
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<_, _>>()?;
        pts.push(Point { features: feats });
    }
    if pts.is_empty() {
        return Err("CSV is empty".into());
    }
    Ok(pts)
}

/// Однопоточный CPU‑k‑means (для эталона).
fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) -> (Vec<Vec<f32>>, usize) {
    let n = data.len();
    let d = data[0].features.len();
    let mut centroids: Vec<Vec<f32>> = data[..k].iter().map(|p| p.features.clone()).collect();
    let mut asg = vec![0usize; n];
    let mut moved = true;
    let mut it = 0;

    while moved && it < max_it {
        moved = false;
        // assignment
        for (i, p) in data.iter().enumerate() {
            let (mut best_dist, mut best_k) = (f32::MAX, 0);
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist: f32 = c
                    .iter()
                    .zip(&p.features)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = c_idx;
                }
            }
            if asg[i] != best_k {
                moved = true;
                asg[i] = best_k;
            }
        }
        if !moved {
            break;
        }
        // update
        let mut sums = vec![vec![0f32; d]; k];
        let mut counts = vec![0usize; k];
        for (idx, p) in data.iter().enumerate() {
            let c = asg[idx];
            counts[c] += 1;
            for j in 0..d {
                sums[c][j] += p.features[j];
            }
        }
        moved = false;
        for c in 0..k {
            if counts[c] == 0 {
                continue; // пустой кластер
            }
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                sums[c][j] *= inv;
            }
            let delta = sums[c]
                .iter()
                .zip(&centroids[c])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            if delta > eps {
                moved = true;
            }
            centroids[c] = sums[c].clone();
        }
        it += 1;
    }
    (centroids, it)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data = read_points(&args.file)?;
    let n = data.len();
    let d = data[0].features.len();
    println!("Points: {n}, dim: {d}, K: {}", args.k);

    // ===== CPU reference =====
    if !args.opencl {
        let t = Instant::now();
        let (_, iters) = kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU done in {iters} iters, {:?}", t.elapsed());
        return Ok(());
    }

    // ===== GPU INITIALISATION =====
    let flat_data: Vec<f32> = data.iter().flat_map(|p| &p.features).copied().collect();
    let mut centroids: Vec<f32> = flat_data[..args.k * d].to_vec();

    // Kernel (OpenCL 1.2, без атомиков)
    let src = format!(
        "#pragma OPENCL VERSION 120\n\
// fallback for FLT_MAX
#define FLT_MAX 3.402823466e+38F
__kernel void assign_points(
    __global const float* data,
    __global const float* centroids,
    __global uint* assign,
    const uint D,
    const uint K)
{{
    uint gid = get_global_id(0);
    float best = FLT_MAX;
    uint best_k = 0u;
    for (uint c = 0u; c < K; ++c) {{
        float dist = 0.0f;
        for (uint j = 0u; j < D; ++j) {{
            float diff = data[gid * D + j] - centroids[c * D + j];
            dist += diff * diff;
        }}
        if (dist < best) {{ best = dist; best_k = c; }}
    }}
    assign[gid] = best_k;
}}",
    );

    // Build ProQue without дополнительных builder‑функций: совместимо с ocl 0.30
    let pro_que = ProQue::builder().src(src).dims(n).build()?;

    // Buffers
    let data_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(flat_data.len())
        .copy_host_slice(&flat_data)
        .build()?;
    let mut cent_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(centroids.len())
        .copy_host_slice(&centroids)
        .build()?;
    let asg_buf = Buffer::<u32>::builder()
        .queue(pro_que.queue().clone())
        .len(n)
        .build()?;

    let mut assign_kernel = pro_que
        .kernel_builder("assign_points")
        .arg(&data_buf)
        .arg(&cent_buf)
        .arg(&asg_buf)
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    // ===== MAIN ITERATION =====
    let t0 = Instant::now();
    let mut assignments = vec![0u32; n];
    let mut it = 0usize;
    loop {
        it += 1;
        unsafe { assign_kernel.enq()?; }
        asg_buf.read(&mut assignments).enq()?;

        // ---- Update step on CPU ----
        let mut sums = vec![vec![0f32; d]; args.k];
        let mut counts = vec![0usize; args.k];
        for (idx, &cid) in assignments.iter().enumerate() {
            counts[cid as usize] += 1;
            let feats = &data[idx].features;
            for j in 0..d {
                sums[cid as usize][j] += feats[j];
            }
        }
        let mut max_delta = 0f32;
        for c in 0..args.k {
            if counts[c] == 0 {
                continue; // пустой кластер
            }
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                sums[c][j] *= inv;
            }
            // delta
            for j in 0..d {
                let delta = (sums[c][j] - centroids[c * d + j]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                centroids[c * d + j] = sums[c][j];
            }
        }
        // ---- копируем обновлённые центроиды в GPU‑буфер ----
        cent_buf.write(&centroids).enq()?;

        if max_delta < args.eps || it >= args.iterations {
            println!(
                "Converged in {it} iterations, GPU time {:?}",
                t0.elapsed()
            );
            break;
        }
    }
    Ok(())
}