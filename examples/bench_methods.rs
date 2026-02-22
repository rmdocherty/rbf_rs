use std::process::Command;

use image::ImageReader;
use libblur::{
    BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy, fast_bilateral_filter_f32,
};

use rbf_rs::rbf;

fn main() -> std::io::Result<()> {
    let in_dir = "tests/data/blobs.jpg";
    let sigma_spatial = 0.03;
    let sigma_range = 0.1;
    let n: u32 = 20;
    run_rbf(in_dir, sigma_spatial, sigma_range, n)?;
    run_libblur(in_dir, 21, 1.0, 1.0, n)?;
    run_py(in_dir, 21, 75.0, 75.0, n)?;
    run_cpp(in_dir, sigma_range, sigma_spatial, n)?;
    Ok(())
}

fn run_rbf(in_dir: &str, sigma_spatial: f32, sigma_range: f32, n: u32) -> std::io::Result<()> {
    let img_path = in_dir;
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

    let mut buf = rbf::ExternalBuffer::new(width, height, 4, 3);
    let start_time = std::time::Instant::now();
    for _ in 0..n {
        rbf::recursive_bilateral_filter::<4, 3>(
            &signal,
            &guidance,
            width,
            height,
            sigma_spatial,
            sigma_range,
            Some(&mut buf),
        );
    }
    let end_time = std::time::Instant::now();
    println!("({width}, {height}, 3), sigma_range: {sigma_range}, sigma_spatial: {sigma_spatial}");
    println!(
        "RS RBF: {:.6}s",
        (end_time - start_time).as_secs_f32() / n as f32
    );

    Ok(())
}

fn run_libblur(
    in_dir: &str,
    k: u32,
    sigma_spatial: f32,
    sigma_range: f32,
    n: u32,
) -> std::io::Result<()> {
    let img_path = in_dir;
    let img = ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb32f();
    let (width, height) = img.dimensions();

    let src_bytes = img.into_raw();

    let cvt = BlurImage::borrow(&src_bytes, width, height, FastBlurChannels::Channels3);
    let image = cvt
        .linearize(libblur::TransferFunction::Srgb, true)
        .unwrap();

    let mut dst_image = BlurImageMut::default();
    let start_time = std::time::Instant::now();
    for _ in 0..n {
        fast_bilateral_filter_f32(
            &image,
            &mut dst_image,
            k,
            sigma_spatial,
            sigma_range,
            ThreadingPolicy::Adaptive,
        )
        .unwrap();
    }
    let end_time = std::time::Instant::now();
    println!(
        "({width}, {height}, 3), k: {k}, sigma_range: {sigma_range}, sigma_spatial: {sigma_spatial}"
    );
    println!(
        "LB BF: {:.6}s",
        (end_time - start_time).as_secs_f32() / n as f32
    );
    Ok(())
}

fn run_py(infile: &str, k: i32, sigma_color: f32, sigma_space: f32, n: u32) -> std::io::Result<()> {
    let k_str = k.to_string();
    let sigma_color_str = sigma_color.to_string();
    let sigma_space_str = sigma_space.to_string();
    let n_str = (n).to_string();

    let args = vec![
        "run",
        "benches/bench.py",
        "-i",
        infile,
        "-k",
        &k_str,
        "--sigmaColor",
        &sigma_color_str,
        "--sigmaSpace",
        &sigma_space_str,
        "--bench",
        &n_str,
    ];
    let status = Command::new("uv").args(&args).status()?;
    if !status.success() {
        eprintln!("bench.py failed with status: {}", status);
    }
    Ok(())
}

fn run_cpp(infile: &str, sigma_color: f32, sigma_space: f32, n: u32) -> std::io::Result<()> {
    let sigma_color_str = sigma_color.to_string();
    let sigma_space_str = sigma_space.to_string();
    let n_str = (n).to_string();

    let args = vec![
        "nowrite",
        infile,
        &sigma_space_str,
        &sigma_color_str,
        &n_str,
    ];
    let status = Command::new("benches/rbf_cpp_bench").args(&args).status()?;
    if !status.success() {
        eprintln!("bench.py failed with status: {}", status);
    }
    Ok(())
}
