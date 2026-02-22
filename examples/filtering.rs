use image::{ImageReader, RgbImage};
use std::env;

use rbf_rs::rbf;

fn run_rbf(
    in_dir: &str,
    out_dir: &str,
    sigma_spatial: f32,
    sigma_range: f32,
) -> std::io::Result<()> {
    let img_path = in_dir;
    let img = ImageReader::open(img_path)
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
    );

    let filtered_u8 = filtered
        .iter()
        .map(|&v| (v).clamp(0.0, 255.0) as u8)
        .collect::<Vec<u8>>();

    let out_img = RgbImage::from_raw(width as u32, height as u32, filtered_u8)
        .expect("Failed to create output image from filtered data");
    std::fs::create_dir_all("tests/out/compare").unwrap();
    out_img
        .save(out_dir)
        .expect("Failed to save filtered image");

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!(
            "Usage: {} <input_image> <output_image> <sigma_spatial> <sigma_range>",
            args[0]
        );
        std::process::exit(1);
    }

    let in_dir = &args[1];
    let out_dir = &args[2];
    let sigma_spatial: f32 = args[3].parse().expect("Invalid sigma");
    let sigma_range: f32 = args[4].parse().expect("Invalid sigma");

    run_rbf(in_dir, out_dir, sigma_spatial, sigma_range).expect("Failed to run RBF");
}
