use criterion::{Criterion, criterion_group, criterion_main};
use image::ImageReader;

use std::hint::black_box;

use rbf_rs::rbf;

fn bench_rbf_horizontal_pass(c: &mut Criterion) {
    // Load the image as RGB8
    let img_path = "tests/data/blobs.jpg";
    let img = image::ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Prepare signal and guidance buffers
    let signal: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32))
        .collect();
    let guidance: Vec<u8> = img.pixels().flat_map(|p| p.0.iter().copied()).collect();

    // Parameters
    let sigma_spatial = 0.03;
    let sigma_range = 0.1;

    // Prepare normalization buffer
    let n_ch_k = 4; // For RGB + normalization
    let mut buf_a = vec![0.0f32; width * height * n_ch_k];
    let mut buf_b = vec![0.0f32; width * height * n_ch_k];

    // Pad signal with weights
    rbf_rs::rbf::pad_signal_with_weights::<4>(&signal, &mut buf_a, width, height, 1.0);

    // Precompute colour weight table
    let e = 2.71828182845904523536028747135266250_f32;
    let n_colours = 256;
    let inverse_sigma_for_range_table = 1.0 / (sigma_range * n_colours as f32);
    let colour_dist_weight_table: [f32; 256] = (0..n_colours)
        .map(|i| e.powf(-(i as f32) * inverse_sigma_for_range_table))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let alpha_h = e.powf(-f32::sqrt(2.0) / (sigma_spatial * (width as f32)));

    let mut group = c.benchmark_group("rbf_horizontal_pass");
    group.sample_size(10);

    group.bench_function("rbf_horizontal_pass", |b| {
        b.iter(|| {
            rbf_rs::rbf::rbf_horizontal_parallel::<4, 3>(
                black_box(&buf_a),
                black_box(&guidance),
                black_box(&mut buf_b),
                black_box(width),
                black_box(alpha_h),
                black_box(&colour_dist_weight_table),
            );
        });
    });
}

fn bench_transpose_tiled_hwc(c: &mut Criterion) {
    // Load the image as RGB8
    let img_path = "tests/data/blobs.jpg";
    let img = image::ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Prepare guidance buffer
    let guidance: Vec<u8> = img.pixels().flat_map(|p| p.0.iter().copied()).collect();
    let mut buf_img = vec![0u8; width * height * 3];

    let mut group = c.benchmark_group("transpose_tiled_hwc");
    group.sample_size(10);

    group.bench_function("transpose_tiled_hwc", |b| {
        b.iter(|| {
            rbf_rs::rbf::transpose_tiled_hwc::<u8, 3>(
                black_box(&guidance),
                black_box(&mut buf_img),
                black_box(height),
                black_box(width),
            );
        });
    });
}

fn bench_normalize(c: &mut Criterion) {
    // Load the image as RGB8
    let img_path = "tests/data/blobs.jpg";
    let img = image::ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Prepare signal buffer
    let signal: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32))
        .collect();

    // Prepare normalization buffer
    let n_ch_k = 4; // For RGB + normalization
    let mut buf_a = vec![0.0f32; width * height * n_ch_k];
    let mut out_buf = vec![0.0f32; width * height * (n_ch_k - 1)];

    // Pad signal with weights
    rbf_rs::rbf::pad_signal_with_weights::<4>(&signal, &mut buf_a, width, height, 1.0);

    let mut group = c.benchmark_group("normalize");
    group.sample_size(10);

    group.bench_function("normalize", |b| {
        b.iter(|| {
            rbf_rs::rbf::normalize::<4>(black_box(&buf_a), black_box(&mut out_buf));
        });
    });
}

fn bench_rbf_rgb_518(c: &mut Criterion) {
    // Load the image as RGB8
    let img_path = "tests/data/blobs.jpg";
    let img = ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Prepare signal and guidance buffers
    let signal: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32))
        .collect();
    let guidance: Vec<u8> = img.pixels().flat_map(|p| p.0.iter().copied()).collect();

    // Parameters
    let sigma_spatial = 0.03;
    let sigma_range = 0.1;

    let mut buf = rbf::ExternalBuffer::new(width, height, 4, 3);

    let mut group = c.benchmark_group("recursive_bilateral_filter_rgb");
    group.sample_size(10);

    group.bench_function("recursive_bilateral_filter_rgb", |b| {
        b.iter(|| {
            let _filtered = rbf::recursive_bilateral_filter::<4, 3>(
                black_box(&signal),
                black_box(&guidance),
                black_box(width),
                black_box(height),
                black_box(sigma_spatial),
                black_box(sigma_range),
                Some(&mut buf),
            );
        });
    });
}

criterion_group!(
    benches,
    bench_rbf_horizontal_pass,
    bench_transpose_tiled_hwc,
    bench_normalize,
    bench_rbf_rgb_518,
);
criterion_main!(benches);
