//! GPU k‑means (полностью параллельный, OpenCL 1.2‑friendly)
//! CSV: только признаки. Итог центроидов → <input>_centroids.csv

use clap::Parser;
use csv::ReaderBuilder;
use ocl::{Buffer, ProQue};
use std::{error::Error, fs::File, io::Write, path::Path, time::Instant};

#[derive(Clone)]
struct Point {
    features: Vec<f32>,
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(short = 'f', long)]
    file: String,
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    #[clap(short = 'o', long)]
    opencl: bool,
}

fn read_csv<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(File::open(p)?);
    let mut v = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let feats: Vec<f32> = rec.iter().map(|s| s.parse().unwrap()).collect();
        v.push(Point { features: feats });
    }
    Ok(v)
}

const SRC: &str = r#"__kernel void assign_points(
        __global const float* data,
        __global const float* centroids,
        __global uint* assign,
        const uint N, const uint D, const uint K)
{
    uint gid = get_global_id(0);
    if (gid >= N) return;
    float best = FLT_MAX; uint best_k = 0;
    for (uint c = 0; c < K; ++c) {
        float dist = 0.0f;
        for (uint j = 0; j < D; ++j) {
            float diff = data[gid*D + j] - centroids[c*D + j];
            dist += diff * diff;
        }
        if (dist < best) { best = dist; best_k = c; }
    }
    assign[gid] = best_k;
}"#;

fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) -> (Vec<Vec<f32>>, usize) {
    let n = data.len();
    let d = data[0].features.len();
    let mut centroids: Vec<Vec<f32>> = data[..k].iter().map(|p| p.features.clone()).collect();
    let mut assign = vec![0usize; n];

    for it in 0..max_it {
        let mut moved = false;
        for (i, p) in data.iter().enumerate() {
            let mut best = (f32::MAX, 0);
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = c
                    .iter()
                    .zip(&p.features)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                if dist < best.0 {
                    best = (dist, c_idx);
                }
            }
            if assign[i] != best.1 {
                moved = true;
            }
            assign[i] = best.1;
        }
        if !moved {
            return (centroids, it);
        }

        let mut sums = vec![vec![0f32; d]; k];
        let mut cnt = vec![0u32; k];
        for (idx, p) in data.iter().enumerate() {
            let c = assign[idx];
            cnt[c] += 1;
            for j in 0..d {
                sums[c][j] += p.features[j];
            }
        }
        let mut max_delta: f32 = 0.0;
        for c in 0..k {
            if cnt[c] == 0 {
                continue;
            }
            let inv = 1.0 / cnt[c] as f32;
            for j in 0..d {
                let newv = sums[c][j] * inv;
                max_delta = max_delta.max((newv - centroids[c][j]).abs());
                centroids[c][j] = newv;
            }
        }
        if max_delta < eps {
            return (centroids, it);
        }
    }
    (centroids, max_it)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data = read_csv(&args.file)?;
    let n = data.len();
    let d = data[0].features.len();

    if !args.opencl {
        let t = Instant::now();
        let (_, it) = kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU iters: {it}, time: {:?}", t.elapsed());
        return Ok(());
    }

    // flatten
    let flat: Vec<f32> = data.iter().flat_map(|p| &p.features).copied().collect();
    let mut centroids: Vec<f32> = flat[..args.k * d].to_vec();

    let pro_que = ProQue::builder().src(SRC).dims(n).build()?;

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

    let mut assign_kernel = pro_que
        .kernel_builder("assign_points")
        .arg(&data_buf)
        .arg(&cent_buf)
        .arg(&assign_buf)
        .arg(&(n as u32))
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    let t0 = Instant::now();
    for it in 0..args.iterations {
        unsafe {
            assign_kernel.enq()?;
        }
        pro_que.queue().finish()?;

        // read assignments
        let mut assign = vec![0u32; n];
        assign_buf.read(&mut assign).enq()?;

        // CPU reduce
        let mut sums = vec![vec![0f32; d]; args.k];
        let mut cnt = vec![0u32; args.k];
        for (idx, a) in assign.iter().enumerate() {
            let c = *a as usize;
            cnt[c] += 1;
            for j in 0..d {
                sums[c][j] += flat[idx * d + j];
            }
        }
        let mut max_delta: f32 = 0.0;
        for c in 0..args.k {
            if cnt[c] == 0 {
                continue;
            }
            let inv = 1.0 / cnt[c] as f32;
            for j in 0..d {
                let newv = sums[c][j] * inv;
                max_delta = max_delta.max((newv - centroids[c * d + j]).abs());
                centroids[c * d + j] = newv;
            }
        }
        cent_buf.write(&centroids).enq()?;
        if max_delta < args.eps {
            println!("Converged in {it} iters, GPU time {:?}", t0.elapsed());
            break;
        }
    }
    println!("Total GPU time: {:?}", t0.elapsed());

    // save centroids
    let out = args.file.replace(".csv", "_centroids.csv");
    let mut w = File::create(&out)?;
    for c in 0..args.k {
        for j in 0..d {
            write!(
                w,
                "{}{}",
                centroids[c * d + j],
                if j + 1 == d { "\n" } else { "," }
            )?;
        }
    }
    println!("Centroids saved to {out}");
    Ok(())
}
