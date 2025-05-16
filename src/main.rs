//! GPU k‑means (assignment on OpenCL, update на CPU, тип данных f32)
//! CSV содержит только признаки (без меток).
//! После завершения центроиды сохраняются в <input>_centroids.csv,
//! чтобы Python‑бенчмарк мог нарисовать итоговую картинку.

use std::{error::Error, fs::File, io::Write, path::Path, time::Instant};
use clap::Parser;
use csv::ReaderBuilder;
use ocl::{builders::KernelBuilder, Buffer, ProQue};

/// Точка: только признаки
#[derive(Debug, Clone)]
struct Point {
    features: Vec<f32>,
}

#[derive(Parser, Debug)]
struct Args {
    /// CSV‑файл с признаками
    #[clap(short = 'f', long)]
    file: String,
    /// Число кластеров
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Макс. итераций
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    /// Запускать GPU‑вариант
    #[clap(short = 'o', long)]
    opencl: bool,
}

/// CSV → Vec<Point>
fn read_points<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(File::open(p)?);
    let mut pts = Vec::<Point>::new();
    for rec in rdr.records() {
        let rec = rec?;
        let feats: Vec<f32> = rec.iter().map(|s| s.trim().parse::<f32>()).collect::<Result<_, _>>()?;
        pts.push(Point { features: feats });
    }
    Ok(pts)
}

/// CPU‑k‑means (однопоточно) для сравнения
fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) -> (Vec<Vec<f32>>, usize) {
    let n = data.len();
    let d = data[0].features.len();
    let mut centroids: Vec<Vec<f32>> = data[..k].iter().map(|p| p.features.clone()).collect();
    let mut assign = vec![0usize; n];
    let mut sums = vec![vec![0f32; d]; k];
    let mut cnt = vec![0usize; k];

    for it in 0..max_it {
        let mut moved = false;
        // assignment
        for (idx, p) in data.iter().enumerate() {
            let (mut best_d, mut best_k) = (f32::MAX, 0usize);
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist: f32 = c
                    .iter()
                    .zip(&p.features)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if dist < best_d {
                    best_d = dist;
                    best_k = c_idx;
                }
            }
            if assign[idx] != best_k {
                moved = true;
                assign[idx] = best_k;
            }
        }
        if !moved {
            return (centroids, it);
        }
        // zero accumulators
        for v in &mut sums {
            v.fill(0.0);
        }
        cnt.fill(0);
        // accumulate
        for (p_idx, p) in data.iter().enumerate() {
            let c = assign[p_idx];
            cnt[c] += 1;
            for j in 0..d {
                sums[c][j] += p.features[j];
            }
        }
        // update centroids
        moved = false;
        for c in 0..k {
            if cnt[c] == 0 {
                continue; // пустой кластер
            }
            let inv = 1.0 / (cnt[c] as f32);
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
        if !moved {
            return (centroids, it);
        }
    }
    (centroids, max_it)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data = read_points(&args.file)?;
    let n = data.len();
    let d = data[0].features.len();
    println!("Points: {n}, dim: {d}, K: {}", args.k);

    // helper to save centroids => CSV
    let save_centroids = |buf: &[f32]| -> Result<(), Box<dyn Error>> {
        let path = if args.file.ends_with(".csv") {
            format!("{}{}_centroids.csv", &args.file[..args.file.len() - 4], "")
        } else {
            format!("{}_centroids.csv", &args.file)
        };
        let mut w = File::create(&path)?;
        for c in 0..args.k {
            let start = c * d;
            let line = buf[start..start + d]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(w, "{}", line)?;
        }
        println!("Centroids saved → {path}");
        Ok(())
    };

    if !args.opencl {
        let t = Instant::now();
        let (c_final, iters) = kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU done in {iters} iterations, {:?}", t.elapsed());
        // блюрим в 1‑D массив для единообразия
        let flat: Vec<f32> = c_final.into_iter().flat_map(|v| v.into_iter()).collect();
        save_centroids(&flat)?;
        return Ok(());
    }

    // -------- GPU PART (assignment only) --------
    // Flatten data once
    let flat_data: Vec<f32> = data.iter().flat_map(|p| p.features.iter()).copied().collect();
    let mut centroids: Vec<f32> = flat_data[..args.k * d].to_vec(); // первые K точек

    // OpenCL kernels (требует только 1.2)
    const SRC: &str = r#"    
    __kernel void assign_points(
        __global const float* data,   // N×D
        __global const float* centroids,
        __global uint* assign,
        const uint D, const uint K)
    {
        uint gid = get_global_id(0);
        float best = FLT_MAX;
        uint best_k = 0;
        for(uint c = 0; c < K; ++c){
            float dist = 0.0f;
            for(uint j = 0; j < D; ++j){
                float diff = data[gid*D + j] - centroids[c*D + j];
                dist += diff * diff;
            }
            if(dist < best){ best = dist; best_k = c; }
        }
        assign[gid] = best_k;
    }
    "#;

    let wg = 64usize;
    let global = ((n + wg - 1) / wg) * wg;
    let pro_que = ProQue::builder().src(SRC).dims(global).build()?;

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

    let mut assign_kernel = KernelBuilder::new()
        .program(&pro_que.program())
        .name("assign_points")
        .queue(pro_que.queue().clone())
        .global_work_size(global)
        .local_work_size(wg)
        .arg(&data_buf)
        .arg(&cent_buf)
        .arg(&asg_buf)
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    let mut assign = vec![0u32; n];
    let t0 = Instant::now();
    for it in 0..args.iterations {
        unsafe { assign_kernel.enq()?; }
        asg_buf.read(&mut assign).enq()?;

        // ---- пересчёт центроидов на CPU ----
        let mut sums = vec![vec![0f32; d]; args.k];
        let mut cnt = vec![0usize; args.k];
        for (idx, &c) in assign.iter().enumerate() {
            let cid = c as usize;
            cnt[cid] += 1;
            let base = idx * d;
            for j in 0..d {
                sums[cid][j] += flat_data[base + j];
            }
        }
        let mut moved = false;
        for c in 0..args.k {
            if cnt[c] == 0 { continue; }
            let inv = 1.0 / (cnt[c] as f32);
            for j in 0..d { sums[c][j] *= inv; }
            let delta = sums[c]
                .iter()
                .zip(centroids[c * d..(c + 1) * d].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            if delta > args.eps { moved = true; }
            // copy back
            for j in 0..d { centroids[c * d + j] = sums[c][j]; }
        }
        cent_buf.write(&centroids).enq()?;
        if !moved {
            println!("Converged in {it} iterations, {:?}", t0.elapsed());
            break;
        }
    }
    println!("Total GPU time: {:?}", t0.elapsed());
    save_centroids(&centroids)?;
    Ok(())
}
