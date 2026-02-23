use std::process::Command;

use image::ImageReader;

use rbf_rs::rbf;

fn run_cpp(infile: &str, outfile: &str, sigma_color: f32, sigma_space: f32) -> std::io::Result<()> {
    let sigma_color_str = sigma_color.to_string();
    let sigma_space_str = sigma_space.to_string();
    let n_str = (1_u8).to_string();

    let args = vec![outfile, infile, &sigma_space_str, &sigma_color_str, &n_str];
    let status = Command::new("benches/rbf_cpp_bench").args(&args).status()?;
    if !status.success() {
        eprintln!("bench.py failed with status: {}", status);
    }
    Ok(())
}

/// Run our implentation and compare to reference C++ implementation on test image
#[test]
fn test() {
    let infile = "tests/data/blobs.jpg";
    let oufile = "tests/out/tmp_blobs_cpp.png";

    let sigma_spatial = 0.03;
    let sigma_range = 0.1;

    run_cpp(infile, oufile, sigma_range, sigma_spatial).unwrap();
    let cpp_filtered_img = ImageReader::open(oufile)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8();

    let cpp_filtered_buf: Vec<u8> = cpp_filtered_img.to_vec();

    let img = ImageReader::open(infile)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .to_rgb32f();

    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    // For filtering signal == guidance
    let signal: Vec<f32> = img.to_vec().iter().map(|v| 255.0 * v).collect();
    let guidance: Vec<u8> = signal.iter().map(|&v| v as u8).collect();

    let filtered = rbf::recursive_bilateral_filter::<4, 3>(
        &signal,
        &guidance,
        width,
        height,
        sigma_spatial,
        sigma_range,
        None,
    );

    let mut sum_diff = 0.0;
    for i in 0..filtered.len() {
        let cpp_val = cpp_filtered_buf[i];
        let rbf_val = filtered[i] as u8;
        sum_diff += (cpp_val as f32 - rbf_val as f32).abs();
    }
    let mean_diff = sum_diff / filtered.len() as f32;
    const THRESHOLD: f32 = 1.0;
    assert!(
        mean_diff < THRESHOLD,
        "Mean absolute difference between C++ and Rust outputs is too high: {}",
        mean_diff
    );
}
