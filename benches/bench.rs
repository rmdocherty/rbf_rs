use criterion::{Criterion, criterion_group, criterion_main};
use image::{ImageReader, RgbImage};

use std::hint::black_box;

use rbf_rs::rbf;

fn bench_recursive_bilateral_filter(c: &mut Criterion) {
    // Load the image as RGB8
    let img_path = "tests/data/img.jpg";
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
            );
        });
    });

    let img_l_path = "tests/data/img_l.jpg";
    let img_l = ImageReader::open(img_l_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_luma8();

    // Prepare signal and guidance buffers
    let signal_l: Vec<f32> = img_l
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32))
        .collect();
    let guidance_l: Vec<u8> = img_l.pixels().flat_map(|p| p.0.iter().copied()).collect();

    group.bench_function("recursive_bilateral_filter_l", |b| {
        b.iter(|| {
            let _filtered = rbf::recursive_bilateral_filter::<2, 1>(
                black_box(&signal_l),
                black_box(&guidance_l),
                black_box(width),
                black_box(height),
                black_box(sigma_spatial),
                black_box(sigma_range),
            );
        });
    });

    // Run filter once and save output
    let filtered = rbf::recursive_bilateral_filter::<4, 3>(
        &signal,
        &guidance,
        width,
        height,
        sigma_spatial,
        sigma_range,
    );

    let filtered_u8 = filtered
        .iter()
        .map(|&v| (v).clamp(0.0, 255.0) as u8)
        .collect::<Vec<u8>>();

    let avg = filtered.iter().copied().sum::<f32>() / filtered.len() as f32;
    println!("Average filtered value: {}", avg);

    let out_img = RgbImage::from_raw(width as u32, height as u32, filtered_u8)
        .expect("Failed to create output image from filtered data");
    std::fs::create_dir_all("tests/out").unwrap();
    out_img
        .save("tests/out/img_filtered.png")
        .expect("Failed to save filtered image");
}

criterion_group!(benches, bench_recursive_bilateral_filter);
criterion_main!(benches);
