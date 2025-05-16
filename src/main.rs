//! GPU k‑means (OpenCL 1.2 friendly):
//!   • GPU – assignment step (each work‑item = 1 точка)  
//!   • CPU – update centroids  
//! CSV файл содержит **только признаки**.  
//! После сходимости центроиды сохраняются в `<input>_centroids.csv`.
//! Работает даже под pocl/mesa; избегает atomics и group‑reduce.

use std::{error::Error, fs::File, io::Write, path::Path, time::Instant};
use clap::Parser;
use csv::ReaderBuilder;
use ocl::{builders::KernelBuilder, Buffer, ProQue};

#[derive(Clone)]
struct Point {
    features: Vec<f32>,
}

#[derive(Parser, Debug)]
struct Args {
    /// CSV с признаками
    #[clap(short = 'f', long)]
    file: String,
    /// K кластеров
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Макс. итераций
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    /// Использовать GPU (без флага – только CPU)
    #[clap(short = 'o', long)]
    opencl: bool,
}

/// Читает CSV → Vec<Point> (только float‑признаки)
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

/// CPU k‑means (Baseline / Update для GPU‑варианта)
fn cpu_update(assign: &[u32], data: &[Point], k: usize, d: usize) -> Vec<f32> {
    let mut sums = vec![vec![0f32; d]; k];
    let mut cnt = vec![0u32; k];
    for (idx, a) in assign.iter().enumerate() {
        let c = *a as usize;
        cnt[c] += 1;
        for j in 0..d {
            sums[c][j] += data[idx].features[j];
        }
    }
    // new centroids
    let mut centroids = vec![0f32; k * d];
    for c in 0..k {
        let inv = 1.0 / cnt[c].max(1) as f32; // avoid div 0
        for j in 0..d {
            centroids[c * d + j] = sums[c][j] * inv;
        }
    }
    centroids
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data = read_points(&args.file)?;
    let n = data.len();
    let d = data[0].features.len();
    println!("Points: {n}, dim: {d}, K: {}", args.k);

    if !args.opencl {
        // чисто CPU (для бенча)
        let mut centroids: Vec<f32> = data[..args.k]
            .iter()
            .flat_map(|p| p.features.iter())
            .copied()
            .collect();
        let t0 = Instant::now();
        for it in 0..args.iterations {
            // assignment (CPU)
            let mut assign = vec![0u32; n];
            for (idx, p) in data.iter().enumerate() {
                let mut best = (f32::MAX, 0u32);
                for c in 0..args.k {
                    let dist: f32 = (0..d)
                        .map(|j| {
                            let diff = p.features[j] - centroids[c * d + j];
                            diff * diff
                        })
                        .sum();
                    if dist < best.0 {
                        best = (dist, c as u32);
                    }
                }
                assign[idx] = best.1;
            }
            let new_centroids = cpu_update(&assign, &data, args.k, d);
            let delta = new_centroids
                .iter()
                .zip(&centroids)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            centroids = new_centroids;
            if delta < args.eps {
                println!("Converged in {it} iterations, {:?}", t0.elapsed());
                break;
            }
        }
        return Ok(());
    }

    // ---------- GPU вариант: assignment на GPU ----------
    // Плоский массив признаков
    let flat: Vec<f32> = data.iter().flat_map(|p| p.features.iter()).copied().collect();
    let mut centroids: Vec<f32> = flat[..args.k * d].to_vec(); // init

    // OpenCL
    const SRC: &str = r#"__kernel void assign_points(
        __global const float* data,
        __global const float* cent,
        __global uint* assign,
        const uint N, const uint D, const uint K)
    {
        uint gid = get_global_id(0);
        if (gid >= N) return;
        float best = FLT_MAX;
        uint best_k = 0;
        for (uint c = 0; c < K; ++c) {
            float dist = 0.0f;
            for (uint j = 0; j < D; ++j) {
                float diff = data[gid * D + j] - cent[c * D + j];
                dist += diff * diff;
            }
            if (dist < best) { best = dist; best_k = c; }
        }
        assign[gid] = best_k;
    }"#;

    // Размеры: work‑group = 64; глобальный = кратно 64 ≥ N
    let wg = 64usize;
    let global = ((n + wg - 1) / wg) * wg;

    let pro_que = ProQue::builder().src(SRC).dims(global).build()?;
    let data_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(flat.len())
        .copy_host_slice(&flat)
        .build()?;
    let mut cent_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(centroids.len())
        .copy_host_slice(&centroids)
        .build()?;
    let assign_buf = Buffer::<u32>::builder()
        .queue(pro_que.queue().clone())
        .len(n)
        .build()?;

    let mut kernel = KernelBuilder::new()
        .program(&pro_que.program())
        .name("assign_points")
        .queue(pro_que.queue().clone())
        .global_work_size(global)
        .local_work_size(wg)
        .arg(&data_buf)
        .arg(&cent_buf)
        .arg(&assign_buf)
        .arg(&(n as u32))
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    // ---- основной цикл ----
    let t0 = Instant::now();
    for it in 0..args.iterations {
        unsafe { kernel.enq()?; }
        pro_que.queue().finish()?;

        // читаем assignment
        let mut assign = vec![0u32; n];
        assign_buf.read(&mut assign).enq()?;

        // CPU обновляет центроиды
        let new_centroids = cpu_update(&assign, &data, args.k, d);
        // проверяем сходимость
        let delta = new_centroids
            .iter()
            .zip(&centroids)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        centroids.copy_from_slice(&new_centroids);
        cent_buf.write(&centroids).enq()?;
        if delta < args.eps {
            println!("Converged in {it} iterations, {:?}", t0.elapsed());
            break;
        }
    }
    println!("Total GPU‑loop time: {:?}", t0.elapsed());

    // -------- сохранить центроиды --------
    let out_path = format!("{}_centroids.csv", args.file.trim_end_matches(".csv"));
    let mut f = File::create(&out_path)?;
    for c in 0..args.k {
        for j in 0..d {
            if j > 0 { write!(f, ",")?; }
            write!(f, "{}", centroids[c * d + j])?;
        }
        writeln!(f)?;
    }
    println!("Centroids saved to {out_path}");
    Ok(())
}
