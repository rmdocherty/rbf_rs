use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;

use crate::rbf::pad_signal_with_weights;
///# Parallelized Recursive Bilateral Filter (RBF)
///
///Based on the ["Recursive Bilateral Filtering"](<https://link.springer.com/chapter/10.1007/978-3-642-33718-5_29>) by Q. Yang et al.
///and the following [C++ implementation](<https://github.com/ufoym/recursive-bf>)
///
pub mod rbf;

// #[wasm_bindgen]
// extern "C" {
//     pub fn alert(s: &str);
// }

// #[wasm_bindgen]
// pub fn greet(name: &str) {
//     alert(&format!("Hello, {}!", name));
// }

#[wasm_bindgen]
pub fn bilateral_filter(
    img_buf: &[u8],
    width: usize,
    height: usize,
    sigma_spatial: f32,
    sigma_range: f32,
) -> Vec<u8> {
    let signal_buf: Vec<f32> = img_buf.iter().map(|&v| v as f32).collect();

    let result = rbf::recursive_bilateral_filter::<4, 3>(
        &signal_buf,
        img_buf,
        width,
        height,
        sigma_spatial,
        sigma_range,
        None,
    );

    let mut plus_alpha = vec![0.0_f32; width * height * 4];
    pad_signal_with_weights::<4>(&result, &mut plus_alpha, width, height, 255.0);

    plus_alpha
        .into_iter()
        .map(|v| v.clamp(0.0, 255.0) as u8)
        .collect()
}
