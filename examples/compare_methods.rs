// use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use image::{ImageReader, RgbImage};

use rbf_rs::rbf;

fn main() -> std::io::Result<()> {
    let mut child = Command::new("ping")
        .arg("example.com")
        .stdout(Stdio::piped())
        .spawn()?;

    Ok(())
}

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
