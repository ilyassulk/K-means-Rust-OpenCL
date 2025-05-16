//! GPU k-means (assignment + update on OpenCL, тип данных f32)
use std::{error::Error, time::Instant, path::Path, fs::File, cmp::max};
use clap::Parser;
use csv::ReaderBuilder;
use ocl::{Buffer, ProQue, builders::KernelBuilder, prm::Float4};

#[derive(Debug, Clone)]
struct Point { features: Vec<f32>, label: i32 }

#[derive(Parser, Debug)]
struct Args {
    /// CSV: features...,label
    #[clap(short='f', long)]
    file: String,
    /// K clusters
    #[clap(short='k', long, default_value_t = 8)]
    k: usize,
    /// Макс. итераций
    #[clap(short='i', long="iterations", default_value_t = 100)]
    iterations: usize,
    /// Порог сходимости
    #[clap(short='e', long="eps", default_value_t = 1e-3)]
    eps: f32,
    /// Запускать на GPU
    #[clap(short='o', long)]
    opencl: bool,
}

/// CSV → Vec<Point>
fn read_points<P: AsRef<Path>>(p: P) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(File::open(p)?);
    let mut pts = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let n = rec.len();
        if n < 2 { return Err("row must have ≥2 columns".into()); }
        let feats: Vec<f32> = rec.iter().take(n-1).map(|s| s.parse().unwrap()).collect();
        pts.push(Point{features: feats, label: rec[n-1].parse()?});
    }
    Ok(pts)
}

/// Однопоточный CPU-k-means (для честного сравнения)
fn kmeans_cpu(data:&[Point], k:usize, max_it:usize, eps:f32)->(Vec<Vec<f32>>,usize){
    let n=data.len(); let d=data[0].features.len();
    let mut centroids:Vec<Vec<f32>> = data[..k].iter().map(|p|p.features.clone()).collect();
    let mut asg=vec![0usize;n]; let mut tmpc=vec![vec![0f32;d];k]; let mut cnt=vec![0usize;k];
    for it in 0..max_it {
        let mut moved=false;
        for (i,p) in data.iter().enumerate() {
            let mut best=(f32::MAX,0);
            for (c_idx,c) in centroids.iter().enumerate() {
                let dist=c.iter().zip(&p.features).map(|(a,b)|(a-b).powi(2)).sum::<f32>();
                if dist<best.0 { best=(dist,c_idx); }
            }
            if asg[i]!=best.1 { moved=true; }
            asg[i]=best.1;
        }
        if !moved { return (centroids,it) }
        // zero
        tmpc.iter_mut().for_each(|v| v.fill(0.0));
        cnt.fill(0);
        // accumulate
        for (p_idx,p) in data.iter().enumerate() {
            let c=asg[p_idx]; cnt[c]+=1;
            for j in 0..d { tmpc[c][j]+=p.features[j]; }
        }
        // update
        for c in 0..k {
            let inv=1.0/(cnt[c] as f32);
            for j in 0..d { tmpc[c][j]*=inv; }
            let delta=tmpc[c].iter().zip(&centroids[c])
                        .map(|(a,b)|(a-b).abs()).fold(0./0.,f32::max);
            centroids[c]=tmpc[c].clone();
            if delta>eps { moved=true; }
        }
        if !moved { return (centroids,it) }
    }
    (centroids,max_it)
}

fn main()->Result<(),Box<dyn Error>>{
    let args=Args::parse();
    let data=read_points(&args.file)?;
    let n=data.len(); let d=data[0].features.len();
    println!("Points: {n}, dim: {d}, K: {}",args.k);
    if !args.opencl {
        let t=Instant::now();
        let (_,iters)=kmeans_cpu(&data,args.k,args.iterations,args.eps);
        println!("CPU done in {iters} iters, {:?}",t.elapsed());
        return Ok(())
    }

    // -------------------- GPU PART --------------------
    // Flatten data
    let flat:Vec<f32>=data.iter().flat_map(|p|p.features.iter()).copied().collect();
    // initial centroids = first K points
    let mut centroids:Vec<f32>=flat[..args.k*d].to_vec();

    // kernels
    const SRC:&str=r#"
    // assignment kernel: one work-item per point
    __kernel void assign_points(
        __global const float* data,          // N×D
        __global float* centroids,           // K×D (updated every iter)
        __global uint*  assign,              // N
        __global float* partial_sums,        // groups × K × D
        __global uint*  partial_counts,      // groups × K
        const uint D, const uint K)
    {
        uint gid = get_global_id(0);
        uint lid = get_local_id(0);
        uint gsize=get_global_size(0);
        __local float lcent[/*MAX_K*MAX_D*/ 1024]; // make sure K*D fits!
        // загрузить центроиды в локальную память
        for(uint i=lid; i<K*D; i+=get_local_size(0))
            lcent[i]=centroids[i];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (gid>=gsize) return;
        // --- найти ближайший центроид ---
        float best = FLT_MAX;
        uint best_k = 0;
        for(uint c=0; c<K; ++c){
            float dist=0.0f;
            for(uint j=0;j<D;++j){
                float diff=data[gid*D+j]-lcent[c*D+j];
                dist+=diff*diff;
            }
            if(dist<best){ best=dist; best_k=c; }
        }
        assign[gid]=best_k;

        // локальные аккумуляторы (K×D и K)
        __local float lsum[/*MAX_K*MAX_D*/ 1024];
        __local uint  lcnt[/*MAX_K*/ 128];
        for(uint i=lid;i<K*D;i+=get_local_size(0)) lsum[i]=0.0f;
        for(uint i=lid;i<K;i+=get_local_size(0))   lcnt[i]=0;
        barrier(CLK_LOCAL_MEM_FENCE);

        // атомарно в локальную память
        for(uint j=0;j<D;++j)
            atomic_add(&lsum[best_k*D+j], data[gid*D+j]);
        atomic_inc(&lcnt[best_k]);
        barrier(CLK_LOCAL_MEM_FENCE);

        // нить 0 пишет локальные суммы → глобальные partial_* массивы
        if(lid==0){
            uint grp=get_group_id(0);
            for(uint c=0;c<K;++c){
                uint base = (grp*K+c)*D;
                for(uint j=0;j<D;++j)
                    partial_sums[base+j]=lsum[c*D+j];
                partial_counts[grp*K+c]=lcnt[c];
            }
        }
    }

    // kernel 2: каждая work-group = 1 centroid; суммируем по partial_* и делим
    __kernel void update_centroids(
        __global float* centroids,
        __global float* partial_sums,
        __global uint*  partial_counts,
        const uint groups, const uint D, const uint K)
    {
        uint k=get_global_id(0);
        if(k>=K) return;
        float sum;
        uint cnt=0;
        for(uint g=0; g<groups; ++g){
            cnt += partial_counts[g*K+k];
            for(uint j=0;j<D;++j){
                sum = work_group_reduce_add( partial_sums[(g*K+k)*D+j] );
            }
        }
        for(uint j=0;j<D;++j)
            centroids[k*D+j]=sum/ (float)cnt;
    }"#;

    let wg = 64usize;                              // размер work-group
    let global = ((n+wg-1)/wg)*wg;                 // кратно wg
    let groups = global/wg;

    let pro_que=ProQue::builder().src(SRC).dims(global).build()?;

    let data_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).len(flat.len()).copy_host_slice(&flat).build()?;
    let mut cent_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).len(centroids.len()).copy_host_slice(&centroids).build()?;
    let asg_buf  = Buffer::<u32>::builder().queue(pro_que.queue().clone()).len(n).build()?;
    let part_sum = Buffer::<f32>::builder().queue(pro_que.queue().clone()).len(groups*args.k*d).build()?;
    let part_cnt = Buffer::<u32>::builder().queue(pro_que.queue().clone()).len(groups*args.k).build()?;

    let mut assign_kernel = KernelBuilder::new()
        .program(&pro_que.program())
        .name("assign_points")
        .queue(pro_que.queue().clone())
        .global_work_size(global)
        .local_work_size(wg)
        .arg(&data_buf).arg(&cent_buf).arg(&asg_buf)
        .arg(&part_sum).arg(&part_cnt)
        .arg(&(d as u32)).arg(&(args.k as u32))
        .build()?;

    let mut update_kernel = KernelBuilder::new()
        .program(&pro_que.program())
        .name("update_centroids")
        .queue(pro_que.queue().clone())
        .global_work_size(args.k)
        .arg(&cent_buf).arg(&part_sum).arg(&part_cnt)
        .arg(&(groups as u32)).arg(&(d as u32)).arg(&(args.k as u32))
        .build()?;

    // ---------- основная итерация ----------
    let t0 = Instant::now();
    for it in 0..args.iterations {
        unsafe{ assign_kernel.enq()?; }
        unsafe{ update_kernel.enq()?; }
        pro_que.queue().finish()?;

        // копируем центроиды на хост, чтобы проверить сходимость
        cent_buf.read(&mut centroids).enq()?;
        let delta = centroids.chunks(d)
                      .zip(flat[..args.k*d].chunks(d))
                      .map(|(a,b)| a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0./0.,f32::max))
                      .fold(0./0.,f32::max);
        if delta < args.eps {
            println!("Converged in {it} iterations, {:?}", t0.elapsed());
            break;
        }
        // записываем обновлённые центроиды обратно
        cent_buf.write(&centroids).enq()?;
    }
    println!("Total GPU time: {:?}", t0.elapsed());
    Ok(())
}
