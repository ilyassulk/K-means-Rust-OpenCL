//! GPU k‑means (assignment + update on OpenCL, тип данных f32)
//! Совместим с CSV без столбца меток. Каждая строка — только признаки.

use std::{error::Error, fs::File, path::Path, time::Instant};
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
    /// Путь к CSV с признаками
    #[clap(short = 'f', long)]
    file: String,
    /// Кол‑во кластеров
    #[clap(short = 'k', long, default_value_t = 8)]
    k: usize,
    /// Макс. итераций
    #[clap(short = 'i', long = "iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости L∞ по координатам центроидов
    #[clap(short = 'e', long = "eps", default_value_t = 1e-3)]
    eps: f32,
    /// Включить расчёт на GPU (без флага — CPU)
    #[clap(short = 'o', long)]
    opencl: bool,
}

/// CSV → Vec<Point> (все столбцы — f32, без меток)
fn read_points<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(File::open(p)?);

    let mut pts = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.is_empty() {
            continue; // пропускаем пустые строки
        }
        let feats: Vec<f32> = rec
            .iter()
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<_, _>>()?;
        pts.push(Point { features: feats });
    }
    if pts.is_empty() {
        return Err("CSV не содержит ни одной точки".into());
    }
    Ok(pts)
}

/// Однопоточный CPU‑k‑means (для сравнения)
fn kmeans_cpu(data: &[Point], k: usize, max_it: usize, eps: f32) -> (Vec<Vec<f32>>, usize) {
    let n = data.len();
    let d = data[0].features.len();

    // Первые K точек — начальные центроиды
    let mut centroids: Vec<Vec<f32>> = data[..k]
        .iter()
        .map(|p| p.features.clone())
        .collect();
    let mut assignments = vec![0usize; n];
    let mut tmp = vec![vec![0f32; d]; k];
    let mut counts = vec![0usize; k];

    for it in 0..max_it {
        let mut moved = false;
        // Шаг назначения
        for (idx, p) in data.iter().enumerate() {
            let (mut best_dist, mut best_k) = (f32::MAX, 0usize);
            for (ci, c) in centroids.iter().enumerate() {
                let dist = c
                    .iter()
                    .zip(&p.features)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ci;
                }
            }
            if assignments[idx] != best_k {
                moved = true;
            }
            assignments[idx] = best_k;
        }
        if !moved {
            return (centroids, it);
        }

        // Обнуление аккумуляторов
        tmp.iter_mut().for_each(|v| v.fill(0.0));
        counts.fill(0);

        // Накопление сумм
        for (idx, p) in data.iter().enumerate() {
            let c = assignments[idx];
            counts[c] += 1;
            for j in 0..d {
                tmp[c][j] += p.features[j];
            }
        }

        // Обновление центроидов
        moved = false;
        for c in 0..k {
            if counts[c] == 0 {
                continue; // пустой кластер — пропускаем
            }
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                tmp[c][j] *= inv;
            }
            let delta = tmp[c]
                .iter()
                .zip(&centroids[c])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            if delta > eps {
                moved = true;
            }
            centroids[c] = tmp[c].clone();
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

    // ---------- CPU ----------
    if !args.opencl {
        let t = Instant::now();
        let (_, iters) = kmeans_cpu(&data, args.k, args.iterations, args.eps);
        println!("CPU done in {iters} iters, {:?}", t.elapsed());
        return Ok(());
    }

    // -------------------- GPU PART --------------------
    // Плоский массив данных size = N×D
    let flat: Vec<f32> = data
        .iter()
        .flat_map(|p| p.features.iter())
        .copied()
        .collect();

    // Начальные центроиды = первые K точек
    let mut centroids: Vec<f32> = flat[..args.k * d].to_vec();

    // OpenCL kernel source
    const SRC: &str = r#"    
    __kernel void assign_points(
        __global const float* data,          // N×D
        __global float* centroids,           // K×D
        __global uint*  assign,              // N
        __global float* partial_sums,        // groups × K × D
        __global uint*  partial_counts,      // groups × K
        const uint D, const uint K)
    {
        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint gsize = get_global_size(0);

        __local float lcent[1024];           // убедитесь, что K*D <= 1024
        for (uint i = lid; i < K * D; i += get_local_size(0))
            lcent[i] = centroids[i];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (gid >= gsize) return;

        // поиск ближайшего центроида
        float best = FLT_MAX;
        uint  best_k = 0;
        for (uint c = 0; c < K; ++c) {
            float dist = 0.0f;
            for (uint j = 0; j < D; ++j) {
                float diff = data[gid * D + j] - lcent[c * D + j];
                dist += diff * diff;
            }
            if (dist < best) { best = dist; best_k = c; }
        }
        assign[gid] = best_k;

        // локальные буферы для частичных сумм
        __local float lsum[1024];
        __local uint  lcnt[128];
        for (uint i = lid; i < K * D; i += get_local_size(0)) lsum[i] = 0.0f;
        for (uint i = lid; i < K;     i += get_local_size(0)) lcnt[i] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        // аккумулируем точку
        for (uint j = 0; j < D; ++j)
            atomic_add(&lsum[best_k * D + j], data[gid * D + j]);
        atomic_inc(&lcnt[best_k]);
        barrier(CLK_LOCAL_MEM_FENCE);

        // нить 0 группы пишет результаты в глобальную память
        if (lid == 0) {
            uint grp = get_group_id(0);
            for (uint c = 0; c < K; ++c) {
                uint base = (grp * K + c) * D;
                for (uint j = 0; j < D; ++j)
                    partial_sums[base + j] = lsum[c * D + j];
                partial_counts[grp * K + c] = lcnt[c];
            }
        }
    }

    __kernel void update_centroids(
        __global float* centroids,
        __global float* partial_sums,
        __global uint*  partial_counts,
        const uint groups, const uint D, const uint K)
    {
        uint k = get_global_id(0);
        if (k >= K) return;
        float sum;
        uint cnt = 0;
        for (uint g = 0; g < groups; ++g) {
            cnt += partial_counts[g * K + k];
            for (uint j = 0; j < D; ++j) {
                sum = work_group_reduce_add(partial_sums[(g * K + k) * D + j]);
            }
        }
        if (cnt == 0) return; // пустой кластер — сохраняем старый центроид
        for (uint j = 0; j < D; ++j)
            centroids[k * D + j] = sum / (float)cnt;
    }"#;

    // Размеры сетки
    let wg = 64usize;                   // work‑group size
    let global = ((n + wg - 1) / wg) * wg; // округляем вверх
    let groups = global / wg;

    let pro_que = ProQue::builder().src(SRC).dims(global).build()?;

    // Буферы
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
    let part_sum_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(groups * args.k * d)
        .build()?;
    let part_cnt_buf = Buffer::<u32>::builder()
        .queue(pro_que.queue().clone())
        .len(groups * args.k)
        .build()?;

    // --- kernels
    let mut assign_kernel = KernelBuilder::new()
        .program(&pro_que.program())
        .name("assign_points")
        .queue(pro_que.queue().clone())
        .global_work_size(global)
        .local_work_size(wg)
        .arg(&data_buf)
        .arg(&cent_buf)
        .arg(&assign_buf)
        .arg(&part_sum_buf)
        .arg(&part_cnt_buf)
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    let mut update_kernel = KernelBuilder::new()
        .program(&pro_que.program())
        .name("update_centroids")
        .queue(pro_que.queue().clone())
        .global_work_size(args.k)
        .arg(&cent_buf)
        .arg(&part_sum_buf)
        .arg(&part_cnt_buf)
        .arg(&(groups as u32))
        .arg(&(d as u32))
        .arg(&(args.k as u32))
        .build()?;

    // ---------- Итерации ----------
    let start = Instant::now();
    let mut prev_centroids = centroids.clone();

    for it in 0..args.iterations {
        unsafe { assign_kernel.enq()?; }
        unsafe { update_kernel.enq()?; }
        pro_que.queue().finish()?;

        // читаем новые центроиды
        cent_buf.read(&mut centroids).enq()?;

        // проверка сходимости
        let delta = centroids
            .iter()
            .zip(&prev_centroids)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        if delta < args.eps {
            println!("Converged in {it} iterations, {:?}", start.elapsed());
            break;
        }
        prev_centroids.copy_from_slice(&centroids);

        // записываем центроиды обратно в буфер
        cent_buf.write(&centroids).enq()?;
    }
    println!("Total GPU time: {:?}", start.elapsed());
    Ok(())
}