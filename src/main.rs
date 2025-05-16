//! GPU k‑means (OpenCL‑GPU assignment, CPU update, f32)
//! Работает на OpenCL 1.2 (pocl, старые драйверы) — без atomics и reduce‑intrinsics.
//! CSV содержит **только признаки** (без меток).
//! По завершении записывает итоговые центроиды в `<input>_centroids.csv`.

use clap::Parser;
use csv::ReaderBuilder;
use ocl::{Buffer, Kernel, ProQue};
use std::{error::Error, fs::File, io::Write, path::Path, time::Instant};

#[derive(Clone)]
struct Point {
    features: Vec<f32>,
}

#[derive(Parser, Debug)]
struct Args {
    /// CSV‑файл с признаками
    #[clap(short = 'f', long)]
    file: String,
    /// K кластеров
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Максимум итераций
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости (макс |Δ| по координате)
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    /// Использовать GPU (иначе — чистый CPU)
    #[clap(short = 'o', long)]
    opencl: bool,
}

/// Чтение CSV без меток
fn read_points<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(File::open(p)?);
    let mut pts = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let feats: Vec<f32> = rec
            .iter()
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<_, _>>()?;
        pts.push(Point { features: feats });
    }
    Ok(pts)
}

/// Чистый CPU‑вариант (для бенча)
fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) -> (Vec<Vec<f32>>, usize) {
    let n = data.len();
    let d = data[0].features.len();
    let mut centroids: Vec<Vec<f32>> = data[..k].iter().map(|p| p.features.clone()).collect();
    let mut assign = vec![0usize; n];

    for it in 0..max_it {
        let mut moved = false;
        // assignment
        for (idx, p) in data.iter().enumerate() {
            let (mut best_dist, mut best_k) = (f32::MAX, 0);
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = c
                    .iter()
                    .zip(&p.features)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = c_idx;
                }
            }
            if assign[idx] != best_k {
                moved = true;
            }
            assign[idx] = best_k;
        }
        if !moved {
            return (centroids, it);
        }
        // update on CPU
        let mut sums = vec![vec![0f32; d]; k];
        let mut cnt = vec![0usize; k];
        for (idx, p) in data.iter().enumerate() {
            let c = assign[idx];
            cnt[c] += 1;
            for j in 0..d {
                sums[c][j] += p.features[j];
            }
        }
        moved = false;
        for c in 0..k {
            if cnt[c] == 0 {
                continue;
            }
            let inv = 1.0 / cnt[c] as f32;
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

    if !args.opencl {
        let t = Instant::now();
        let (_, iters) = kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU done in {iters} iters, {:?}", t.elapsed());
        return Ok(());
    }

    // GPU assignment kernel -------------------------------------------------
    const SRC: &str = r#"
    __kernel void assign_points(
        __global const float* data,
        __global const float* centroids,
        __global uint* assign,
        const uint D, const uint K)
    {
        uint gid = get_global_id(0);
        float best = FLT_MAX;
        uint best_k = 0u;
        for(uint c=0u; c<K; ++c){
            float dist = 0.0f;
            for(uint j=0u; j<D; ++j){
                float diff = data[gid*D + j] - centroids[c*D + j];
                dist += diff*diff;
            }
            if(dist < best){ best = dist; best_k = c; }
        }
        assign[gid] = best_k;
    }"#;

    // подготовка данных
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|p| p.features.iter())
        .copied()
        .collect();
    let mut centroids: Vec<f32> = flat[..args.k * d].to_vec();
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

    let mut kernel = Kernel::builder()
        .program(&pro_que.program())
        .name("assign_points")
        .queue(pro_que.queue().clone())
        .global_work_size(global)
        .local_work_size(wg)
        .arg(&data_buf)
        .arg(&cent_buf)
        .arg(&assign_buf)
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    let mut assignments = vec![0u32; n];
    let t0 = Instant::now();
    for it in 0..args.iterations {
        // write centroids to device
        cent_buf.write(&centroids).enq()?;

        unsafe {
            kernel.enq()?;
        }
        pro_que.queue().finish()?;

        // read assignments back
        assign_buf.read(&mut assignments).enq()?;

        // CPU update phase
        let mut sums = vec![vec![0f32; d]; args.k];
        let mut cnt = vec![0usize; args.k];
        for (idx, &c_idx) in assignments.iter().enumerate() {
            let c = c_idx as usize;
            cnt[c] += 1;
            let feat_slice = &flat[idx * d..(idx + 1) * d];
            for j in 0..d {
                sums[c][j] += feat_slice[j];
            }
        }
        let mut max_delta = 0f32;
        for c in 0..args.k {
            if cnt[c] == 0 {
                continue;
            }
            let inv = 1.0 / cnt[c] as f32;
            for j in 0..d {
                sums[c][j] *= inv;
            }
            let delta = sums[c]
                .iter()
                .zip(&centroids[c * d..(c + 1) * d])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            max_delta = max_delta.max(delta);
            // copy back to flat centroids vec
            for j in 0..d {
                centroids[c * d + j] = sums[c][j];
            }
        }
        if max_delta < args.eps {
            println!("Converged in {it} iterations, {:?}", t0.elapsed());
            break;
        }
    }
    println!("Total GPU time: {:?}", t0.elapsed());

    // ---------- save centroids ----------
    let out_path = format!("{}_centroids.csv", args.file.trim_end_matches(".csv"));
    let mut f = File::create(&out_path)?;
    for c in 0..args.k {
        let row: Vec<String> = centroids[c * d..(c + 1) * d]
            .iter()
            .map(|v| v.to_string())
            .collect();
        writeln!(f, "{}", row.join(","))?;
    }
    println!("Centroids saved to {out_path}");
    Ok(())
}
