//! GPU k‑means, полностью параллельные assignment **и** update steps на OpenCL 1.2
//! • assignment — каждая точка считает расстояния до K центроидов и записывает
//!   ближайший кластер + локальные частичные суммы.
//! • update — отдельное ядро суммирует partial_sums / partial_counts → новые центроиды.
//!   Реализовано без `atomic_add` и intrinsics, совместимо с pocl / старым драйвером.
//! • CSV содержит **только признаки**.  Итоговые центроиды сохраняются в
//!   `<input>_centroids.csv`, скрипт python строит финальный рисунок.
//! • Все вычисления в `f32`.

use std::{error::Error, fs::File, io::Write, path::Path, time::Instant};
use clap::Parser;
use csv::ReaderBuilder;
use ocl::{Buffer, Kernel, ProQue};

#[derive(Clone)]
struct Point { features: Vec<f32> }

#[derive(Parser, Debug)]
struct Args {
    #[clap(short='f', long)] file: String,
    #[clap(short='k', long, default_value_t = 8)] k: usize,
    #[clap(short='i', long="iterations", default_value_t = 100)] iterations: usize,
    #[clap(short='e', long="eps", default_value_t = 1e-3)] eps: f32,
    /// запустить GPU‑вариант
    #[clap(short='o', long)] opencl: bool,
}

fn read_points<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(File::open(p)?);
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let feats: Vec<f32> = rec.iter().map(|s| s.parse::<f32>().unwrap()).collect();
        out.push(Point { features: feats });
    }
    Ok(out)
}

/// CPU‑справка (для сравнения)
fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) {
    let n = data.len();
    let d = data[0].features.len();
    let mut centroids: Vec<Vec<f32>> = data[..k].iter().map(|p| p.features.clone()).collect();
    let mut asg = vec![0usize; n];
    let mut sums = vec![vec![0f32; d]; k];
    let mut cnt = vec![0usize; k];
    for _ in 0..max_it {
        // assignment
        for (idx, p) in data.iter().enumerate() {
            let mut best = (f32::MAX, 0);
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist: f32 = c.iter().zip(&p.features).map(|(a, b)| (a - b) * (a - b)).sum();
                if dist < best.0 {
                    best = (dist, c_idx);
                }
            }
            asg[idx] = best.1;
        }
        // zero
        for v in sums.iter_mut() { v.fill(0.0); }
        cnt.fill(0);
        // accumulate
        for (idx, p) in data.iter().enumerate() {
            let c = asg[idx];
            cnt[c] += 1;
            for j in 0..d { sums[c][j] += p.features[j]; }
        }
        // update
        let mut max_delta = 0.0;
        for c in 0..k {
            if cnt[c] == 0 { continue; }
            let inv = 1.0 / cnt[c] as f32;
            for j in 0..d {
                let newv = sums[c][j] * inv;
                max_delta = max_delta.max((newv - centroids[c][j]).abs());
                centroids[c][j] = newv;
            }
        }
        if max_delta < eps { break; }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data = read_points(&args.file)?;
    assert!(!data.is_empty());
    let n = data.len();
    let d = data[0].features.len();
    println!("N={n}, D={d}, K={} (GPU={})", args.k, args.opencl);

    if !args.opencl {
        let t = Instant::now();
        kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU done in {:?}", t.elapsed());
        return Ok(());
    }

    // ---------- GPU ----------
    let flat: Vec<f32> = data.iter().flat_map(|p| &p.features).copied().collect();
    let mut centroids: Vec<f32> = flat[..args.k * d].to_vec();

    // kernels
    const SRC: &str = r#"    
    __kernel void assign_points(
        __global const float* data,   // N×D
        __global const float* centroids, // K×D
        __global uint* assign,        // N
        __global float* partial_sums, // groups×K×D
        __global uint*  partial_cnts, // groups×K
        uint N, uint D, uint K)
    {
        uint gid = get_global_id(0);
        if (gid >= N) return;
        uint lid = get_local_id(0);
        uint group = get_group_id(0);
        uint lsize = get_local_size(0);

        // загрузить центроиды в локальную память
        __local float lc[2048]; // достаточно для K*D ≤ 2048
        for (uint i = lid; i < K * D; i += lsize)
            lc[i] = centroids[i];
        barrier(CLK_LOCAL_MEM_FENCE);

        // найти ближайший
        uint best_k = 0;
        float best = FLT_MAX;
        for (uint c = 0; c < K; ++c) {
            float dist = 0.0f;
            for (uint j = 0; j < D; ++j) {
                float diff = data[gid * D + j] - lc[c * D + j];
                dist += diff * diff;
            }
            if (dist < best) { best = dist; best_k = c; }
        }
        assign[gid] = best_k;

        // локальные суммы
        __local float lsum[2048];
        __local uint  lcnt[128];
        for (uint i = lid; i < K * D; i += lsize) lsum[i] = 0.0f;
        for (uint i = lid; i < K; i += lsize)   lcnt[i] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = 0; j < D; ++j)
            atomic_add(&lsum[best_k * D + j], data[gid * D + j]);
        atomic_inc(&lcnt[best_k]);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            for (uint c = 0; c < K; ++c) {
                uint base = (group * K + c) * D;
                for (uint j = 0; j < D; ++j)
                    partial_sums[base + j] = lsum[c * D + j];
                partial_cnts[group * K + c] = lcnt[c];
            }
        }
    }

    __kernel void update_centroids(
        __global float* centroids,
        __global const float* partial_sums,
        __global const uint* partial_cnts,
        uint groups, uint D, uint K)
    {
        uint k = get_global_id(0);
        if (k >= K) return;
        uint cnt = 0;
        for (uint g = 0; g < groups; ++g)
            cnt += partial_cnts[g * K + k];
        if (cnt == 0) return;
        float inv = 1.0f / (float)cnt;
        for (uint j = 0; j < D; ++j) {
            float sum = 0.0f;
            for (uint g = 0; g < groups; ++g) {
                sum += partial_sums[(g * K + k) * D + j];
            }
            centroids[k * D + j] = sum * inv;
        }
    }"#;

    let wg = 64usize;
    let global = ((n + wg - 1) / wg) * wg;
    let groups = global / wg;

    let pq = ProQue::builder().src(SRC).dims(global).build()?;
    // буферы
    let data_buf = Buffer::<f32>::builder().queue(pq.queue().clone()).len(flat.len()).copy_host_slice(&flat).build()?;
    let mut cent_buf = Buffer::<f32>::builder().queue(pq.queue().clone()).len(centroids.len()).copy_host_slice(&centroids).build()?;
    let assign_buf = Buffer::<u32>::builder().queue(pq.queue().clone()).len(n).build()?;
    let ps_buf = Buffer::<f32>::builder().queue(pq.queue().clone()).len(groups * args.k * d).build()?;
    let pc_buf = Buffer::<u32>::builder().queue(pq.queue().clone()).len(groups * args.k).build()?;

    let mut assign_kernel = pq.kernel_builder("assign_points")
        .arg(&data_buf).arg(&cent_buf).arg(&assign_buf)
        .arg(&ps_buf).arg(&pc_buf)
        .arg(&(n as u32)).arg(&(d as u32)).arg(&(args.k as u32))
        .global_work_size(global).local_work_size(wg).build()?;

    let mut update_kernel = pq.kernel_builder("update_centroids")
        .arg(&cent_buf).arg(&ps_buf).arg(&pc_buf)
        .arg(&(groups as u32)).arg(&(d as u32)).arg(&(args.k as u32))
        .global_work_size(args.k).build()?;

    let mut host_assign = vec![0u32; n];
    let t0 = Instant::now();
    for it in 0..args.iterations {
        unsafe { assign_kernel.enq()?; }
        unsafe { update_kernel.enq()?; }
        pq.queue().finish()?;

        // проверка сходимости
        cent_buf.read(&mut centroids).enq()?;
        let max_delta = centroids.chunks(d)
            .zip(flat[..args.k * d].chunks(d))
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max))
            .fold(0.0, f32::max);
        if max_delta < args.eps { println!("Converged in {it} iters"); break; }
        cent_buf.write(&centroids).enq()?;
    }
    println!("GPU total time: {:?}", t0.elapsed());

    // save centroids
    let out_path = format!("{}_centroids.csv", &args.file[..args.file.len() - 4]);
    let mut f = File::create(&out_path)?;
    for c in 0..args.k {
        for j in 0..d {
            write!(f, "{}{}", centroids[c * d + j], if j + 1 == d { "\n" } else { "," })?;
        }
    }
    println!("Centroids saved to {out_path}");
    Ok(())
}
