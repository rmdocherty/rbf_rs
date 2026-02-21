use std::process::Command;

use image::{ImageReader, Rgb32FImage, RgbImage, buffer::ConvertBuffer};
use libblur::{
    BlurImage, BlurImageMut, FastBlurChannels, ThreadingPolicy, fast_bilateral_filter_f32,
};

use rbf_rs::rbf;

fn main() -> std::io::Result<()> {
    let in_dir = "tests/data/blobs.jpg";
    let out_dir_rbf = "tests/out/compare/rbf.png";
    let out_dir_lb = "tests/out/compare/lb.png";
    let out_dir_cv = "tests/out/compare/cv.png";
    let out_dir_cpp = "tests/out/compare/cpp.png";
    let sigma_spatial = 0.03;
    let sigma_range = 0.1;
    run_rbf(in_dir, out_dir_rbf, sigma_spatial, sigma_range)?;
    run_libblur(in_dir, out_dir_lb, 21, 1.0, 1.0)?;
    run_py(in_dir, out_dir_cv, 21, 75.0, 75.0)?;
    run_cpp(in_dir, out_dir_cpp, sigma_range, sigma_spatial)?;
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

fn run_libblur(
    in_dir: &str,
    out_dir: &str,
    k: u32,
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

    let src_bytes = img.into_raw();

    let cvt = BlurImage::borrow(&src_bytes, width, height, FastBlurChannels::Channels3);
    let image = cvt
        .linearize(libblur::TransferFunction::Srgb, true)
        .unwrap();

    let mut dst_image = BlurImageMut::default();
    fast_bilateral_filter_f32(
        &image,
        &mut dst_image,
        k,
        sigma_spatial,
        sigma_range,
        ThreadingPolicy::Adaptive,
    )
    .unwrap();
    let out_buf: Vec<f32> = dst_image.data.borrow().to_vec();
    let filtered_u8 = out_buf
        .iter()
        .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
        .collect::<Vec<u8>>();

    let out_img = RgbImage::from_raw(width as u32, height as u32, filtered_u8)
        .expect("Failed to create output image from filtered data");
    std::fs::create_dir_all("tests/out/compare").unwrap();
    out_img
        .save(out_dir)
        .expect("Failed to save filtered image");
    Ok(())
}

fn run_py(
    infile: &str,
    outfile: &str,
    k: i32,
    sigma_color: f32,
    sigma_space: f32,
) -> std::io::Result<()> {
    let k_str = k.to_string();
    let sigma_color_str = sigma_color.to_string();
    let sigma_space_str = sigma_space.to_string();

    let args = vec![
        "run",
        "benches/bench.py",
        "-i",
        infile,
        "-o",
        outfile,
        "-k",
        &k_str,
        "--sigmaColor",
        &sigma_color_str,
        "--sigmaSpace",
        &sigma_space_str,
    ];
    let status = Command::new("uv").args(&args).status()?;
    if !status.success() {
        eprintln!("bench.py failed with status: {}", status);
    }
    Ok(())
}

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
