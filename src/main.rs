//! GPU k‑means (assignment on OpenCL, update on CPU, тип данных f32)
//! CSV содержит только признаки (без меток).

use clap::Parser;
use csv::ReaderBuilder;
use ocl::{builders::KernelBuilder, Buffer, ProQue};
use std::{error::Error, fs::File, path::Path, time::Instant};

#[derive(Debug, Clone)]
struct Point {
    features: Vec<f32>,
}

#[derive(Parser, Debug)]
struct Args {
    /// CSV‑файл с признаками
    #[clap(short = 'f', long)]
    file: String,
    /// Количество кластеров
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Максимум итераций
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    /// Запускать GPU (если не указан, используется чистый CPU)
    #[clap(short = 'o', long)]
    opencl: bool,
}

/// CSV → Vec<Point> (все столбцы — признаки)
fn read_points<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(File::open(p)?);
    let mut pts = Vec::<Point>::new();
    for rec in rdr.records() {
        let rec = rec?;
        let feats: Vec<f32> = rec
            .iter()
            .map(|s| s.parse::<f32>())
            .collect::<Result<_, _>>()?;
        pts.push(Point { features: feats });
    }
    Ok(pts)
}

/// CPU‑k‑means (однопоточный) — для проверки и сравнения
fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) -> (Vec<Vec<f32>>, usize) {
    let n = data.len();
    let d = data[0].features.len();
    let mut centroids: Vec<Vec<f32>> = data[..k].iter().map(|p| p.features.clone()).collect();
    let mut assign = vec![0usize; n];

    for it in 0..max_it {
        // assignment
        let mut changed = false;
        for (idx, p) in data.iter().enumerate() {
            let (mut best_dist, mut best_k) = (f32::MAX, 0usize);
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
            if assign[idx] != best_k {
                changed = true;
                assign[idx] = best_k;
            }
        }
        if !changed {
            return (centroids, it);
        }
        // update on CPU
        let mut sums = vec![vec![0.0f32; d]; k];
        let mut cnts = vec![0usize; k];
        for (idx, p) in data.iter().enumerate() {
            let c = assign[idx];
            cnts[c] += 1;
            for j in 0..d {
                sums[c][j] += p.features[j];
            }
        }
        let mut max_delta = 0.0f32;
        for c in 0..k {
            if cnts[c] == 0 {
                continue; // пустой кластер
            }
            let inv = 1.0 / cnts[c] as f32;
            for j in 0..d {
                sums[c][j] *= inv;
            }
            let delta = sums[c]
                .iter()
                .zip(&centroids[c])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            max_delta = max_delta.max(delta);
            centroids[c] = sums[c].clone();
        }
        if max_delta < eps {
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

    // ---------------- CPU ONLY ----------------
    if !args.opencl {
        let t0 = Instant::now();
        let (_, iters) = kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU done in {iters} iterations, {:?}", t0.elapsed());
        return Ok(());
    }

    // ---------------- GPU ASSIGNMENT ----------------
    // flatten dataset
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|p| p.features.iter())
        .copied()
        .collect();
    let mut centroids: Vec<f32> = flat[..args.k * d].to_vec();
    let mut assign_host = vec![0u32; n];

    // простейший kernel только для шага assignment (без атомиков, без reduce‑builtin)
    const SRC: &str = r#"__kernel void assign_points(
        __global const float* data,
        __global const float* centroids,
        __global uint* assign,
        const uint D,
        const uint K)
    {
        uint gid = get_global_id(0);
        // каждая нить обрабатывает одну точку
        float best = FLT_MAX;
        uint  best_k = 0;
        for (uint c = 0; c < K; ++c) {
            float dist = 0.0f;
            for (uint j = 0; j < D; ++j) {
                float diff = data[gid * D + j] - centroids[c * D + j];
                dist += diff * diff;
            }
            if (dist < best) { best = dist; best_k = c; }
        }
        assign[gid] = best_k;
    }"#;

    let wg = 64usize;
    let global = ((n + wg - 1) / wg) * wg;

    let pro_que = ProQue::builder()
        .src(SRC)
        .dims(global)
        // ограничиваемся OpenCL 1.2
        .prog_bldr(|b| b.cmplt().cpp_define("CL_TARGET_OPENCL_VERSION", "120"))
        .build()?;

    // буферы
    let data_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(flat.len())
        .copy_host_slice(&flat)
        .build()?;
    let cent_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(centroids.len())
        .copy_host_slice(&centroids)
        .build()?;
    let assign_buf = Buffer::<u32>::builder()
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
        .arg(&assign_buf)
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    // ----- основной цикл -----
    let t0 = Instant::now();
    for it in 0..args.iterations {
        // шаг assignment на GPU
        unsafe {
            assign_kernel.enq()?;
        }
        pro_que.queue().finish()?;

        // читаем назначения
        assign_buf.read(&mut assign_host).enq()?;

        // update на CPU
        let mut sums = vec![vec![0.0f32; d]; args.k];
        let mut cnts = vec![0usize; args.k];
        for (idx, p) in data.iter().enumerate() {
            let c = assign_host[idx] as usize;
            cnts[c] += 1;
            for j in 0..d {
                sums[c][j] += p.features[j];
            }
        }
        let mut max_delta = 0.0f32;
        for c in 0..args.k {
            if cnts[c] == 0 {
                continue;
            }
            let inv = 1.0 / cnts[c] as f32;
            for j in 0..d {
                sums[c][j] *= inv;
            }
            let delta = sums[c]
                .iter()
                .zip(centroids[c * d..(c + 1) * d].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            max_delta = max_delta.max(delta);
            // записываем в массив центроидов подряд
            for j in 0..d {
                centroids[c * d + j] = sums[c][j];
            }
        }
        if max_delta < args.eps {
            println!("Converged in {it} iterations, {:?}", t0.elapsed());
            break;
        }
        // передаём обновлённые центроиды на устройство
        cent_buf.write(&centroids).enq()?;
    }

    println!(
        "Total wall time (GPU assignment + CPU update): {:?}",
        t0.elapsed()
    );
    Ok(())
}
