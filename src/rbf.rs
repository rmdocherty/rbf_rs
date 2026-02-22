/// Parallelized Recursive Bilateral Filter
///
/// Based on "Recursive Bilateral Filtering" by Q. Yang et al. (https://link.springer.com/chapter/10.1007/978-3-642-33718-5_29)
/// and the following C++ implementation (https://github.com/ufoym/recursive-bf)
use rayon::prelude::*;

const E: f32 = 2.71828182845904523536028747135266250_f32;
const N_COLOURS: usize = 256;

pub struct ExternalBuffer {
    pub buf_a: Vec<f32>,
    pub buf_b: Vec<f32>,
    pub buf_img: Vec<u8>,
}

impl ExternalBuffer {
    // Preallocate buffers, including two 'ping-pong' buffers (buf_a & buf_b)
    pub fn new(width: usize, height: usize, n_ch_k: usize, n_ch_guidance: usize) -> Self {
        Self {
            buf_a: vec![0.0f32; width * height * n_ch_k],
            buf_b: vec![0.0f32; width * height * n_ch_k],
            buf_img: vec![0u8; width * height * n_ch_guidance],
        }
    }
}

pub fn recursive_bilateral_filter<const N_CH_K: usize, const N_CH_GUIDANCE: usize>(
    signal: &[f32],
    guidance_img: &[u8],
    width: usize,
    height: usize,
    sigma_spatial: f32,
    sigma_range: f32,
    external_buffer: Option<&mut ExternalBuffer>,
) -> Vec<f32> {
    match external_buffer {
        Some(buf) => recursive_bilateral_filter_impl::<N_CH_K, N_CH_GUIDANCE>(
            signal,
            guidance_img,
            width,
            height,
            sigma_spatial,
            sigma_range,
            buf,
        ),
        None => {
            let mut buf = ExternalBuffer::new(width, height, N_CH_K, N_CH_GUIDANCE);
            recursive_bilateral_filter_impl::<N_CH_K, N_CH_GUIDANCE>(
                signal,
                guidance_img,
                width,
                height,
                sigma_spatial,
                sigma_range,
                &mut buf,
            )
        }
    }
}

/// Implementation of recursive bilateral filter.
///
/// Filters pixels in $signal with respect to colour similarity in $guidance.
/// Does the following:
/// - Pad input buffers with required normalization weights (1.0)
/// - Do L->R and R->L horizontal recursive bilateral filtering on $signal
/// - Transpose $signal and $guidance image
/// - Do horizontal recursive bilateral filtering (now T->B, B->T)
/// - Transpose result
/// - Rescale pixels with accumulated normalization weights; return
fn recursive_bilateral_filter_impl<const N_CH_K: usize, const N_CH_GUIDANCE: usize>(
    signal: &[f32],
    guidance_img: &[u8],
    width: usize,
    height: usize,
    sigma_spatial: f32,
    sigma_range: f32,
    external_buffer: &mut ExternalBuffer,
) -> Vec<f32> {
    let mut src = &mut external_buffer.buf_a;
    let mut dst = &mut external_buffer.buf_b;

    pad_signal_with_weights::<N_CH_K>(signal, &mut dst, width, height, 1.0);
    std::mem::swap(&mut src, &mut dst);

    // exponential decaying weight table with falloff defined by $sigma_range
    // i.e, if two pixels are 4 intensity apart, the weight is $colour_weight_range_table[3]
    let inverse_sigma_for_range_table = 1.0 / (sigma_range * N_COLOURS as f32);
    let colour_dist_weight_table: [f32; N_COLOURS] = (0..N_COLOURS)
        .map(|i| E.powf(-(i as f32) * inverse_sigma_for_range_table))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // alpha is spatial weight for filter
    let alpha_h = E.powf(-f32::sqrt(2.0) / (sigma_spatial * (width as f32)));
    rbf_horizontal_parallel::<N_CH_K, N_CH_GUIDANCE>(
        &src,
        guidance_img,
        &mut dst,
        width,
        alpha_h,
        &colour_dist_weight_table,
    );
    std::mem::swap(&mut src, &mut dst);

    transpose_tiled_hwc::<u8, N_CH_GUIDANCE>(
        guidance_img,
        &mut external_buffer.buf_img,
        height,
        width,
    );
    transpose_tiled_hwc::<f32, N_CH_K>(&src, &mut dst, height, width);
    std::mem::swap(&mut src, &mut dst);

    let alpha_v = E.powf(-f32::sqrt(2.0) / (sigma_spatial * (height as f32)));
    rbf_horizontal_parallel::<N_CH_K, N_CH_GUIDANCE>(
        &src,
        &external_buffer.buf_img,
        &mut dst,
        height,
        alpha_v,
        &colour_dist_weight_table,
    );
    std::mem::swap(&mut src, &mut dst);

    transpose_tiled_hwc::<f32, N_CH_K>(&src, &mut dst, width, height);
    std::mem::swap(&mut src, &mut dst);

    let mut out_buf = vec![0.0f32; width * height * (N_CH_K - 1)];
    normalize::<N_CH_K>(&src, &mut out_buf);
    out_buf
}

/// Horizontal pass of rbf
///
/// For each pixel p_i in row:
/// - calc dist between p_i & p_(i-1) in colour space
/// - get combined spatial-colour weight \propto sigma_colour * dist * sigma_space
/// - find filtered_i = (1 / spatial_weight) * signal_i  + spatial-colour weight * filtered_(i-1)
/// - (this propagates signals across regions of similar colour but preserves harsh edges)
/// - also update the normalization weight in the same way
/// - do this in reverse (i.e from R->L), but this time write 0.5 * (filtered_LR_i + filtered_RL_i)
/// (i.e average across directions)
pub fn rbf_horizontal_parallel<const N_CH_K: usize, const N_CH_GUIDANCE: usize>(
    signal_and_norm: &[f32],
    guidance_img: &[u8],
    output_and_norm_buf: &mut [f32],
    width: usize,
    alpha_h: f32,
    colour_dist_weight_table: &[f32; N_COLOURS],
) {
    let inv_alpha_h = 1.0 - alpha_h;
    let signal_row_size = width * N_CH_K;
    let guidance_row_size = width * N_CH_GUIDANCE;

    output_and_norm_buf
        .par_chunks_exact_mut(signal_row_size)
        .zip(signal_and_norm.par_chunks_exact(signal_row_size))
        .zip(guidance_img.par_chunks_exact(guidance_row_size))
        .for_each(
            |((output_and_norm_buf_row, signal_row), guidance_img_row)| {
                // =========== Forward Pass (Left-to-Right) ===========
                let mut prev_guidance_px = &guidance_img_row[0..N_CH_GUIDANCE];
                let mut state = [0.0f32; N_CH_K];
                // Initialize first pixel & state buffer
                for c in 0..N_CH_K {
                    let init_val = signal_row[c];
                    state[c] = init_val;
                    output_and_norm_buf_row[c] = init_val;
                }

                for x in 1..width {
                    let signal_px_idx = x * N_CH_K;
                    let guidance_px_idx = x * N_CH_GUIDANCE;

                    let current_signal_px = &signal_row[signal_px_idx..(signal_px_idx + N_CH_K)];
                    let current_guidance_px =
                        &guidance_img_row[guidance_px_idx..guidance_px_idx + N_CH_GUIDANCE];
                    let colour_dist =
                        calculate_dist::<N_CH_GUIDANCE>(current_guidance_px, prev_guidance_px);
                    let spatial_and_colour_weight =
                        colour_dist_weight_table[colour_dist as usize] * alpha_h;

                    for c in 0..N_CH_K {
                        let filtered_val = inv_alpha_h * (current_signal_px[c])
                            + spatial_and_colour_weight * state[c];
                        output_and_norm_buf_row[signal_px_idx + c] = filtered_val;
                        state[c] = filtered_val;
                    }
                    prev_guidance_px = current_guidance_px;
                }

                // =========== Backward Pass (Right-to-Left) ===========

                let mut prev_guidance_px_rev =
                    &guidance_img_row[(width - 1) * N_CH_GUIDANCE..width * N_CH_GUIDANCE];
                // Initialize last pixel & state buffer
                for c in 0..N_CH_K {
                    let init_val = signal_row[(width - 1) * N_CH_K + c];
                    state[c] = init_val;
                }

                for x_rev in (0..width - 1).rev() {
                    let signal_px_idx = x_rev * N_CH_K;
                    let guidance_px_idx = x_rev * N_CH_GUIDANCE;

                    let current_signal_px = &signal_row[signal_px_idx..(signal_px_idx + N_CH_K)];
                    let current_guidance_px =
                        &guidance_img_row[guidance_px_idx..guidance_px_idx + N_CH_GUIDANCE];
                    let colour_dist =
                        calculate_dist::<N_CH_GUIDANCE>(current_guidance_px, prev_guidance_px_rev);
                    let spatial_and_colour_weight =
                        colour_dist_weight_table[colour_dist as usize] * alpha_h;

                    for c in 0..N_CH_K {
                        let filtered_val = inv_alpha_h * (current_signal_px[c])
                            + spatial_and_colour_weight * state[c];
                        let filtered_val_fwd = output_and_norm_buf_row[signal_px_idx + c];
                        output_and_norm_buf_row[signal_px_idx + c] =
                            0.5 * (filtered_val + filtered_val_fwd);
                        state[c] = filtered_val;
                    }
                    prev_guidance_px_rev = current_guidance_px;
                }
            },
        );
}

#[inline(always)]
fn calculate_dist<const N_CH_GUIDANCE: usize>(curr: &[u8], prev: &[u8]) -> i32 {
    match N_CH_GUIDANCE {
        // greyscale images
        1 => (curr[0] as i32 - prev[0] as i32).abs(),
        // rgb images
        3 => {
            let dr = (curr[0] as i32 - prev[0] as i32).abs();
            let dg = (curr[1] as i32 - prev[1] as i32).abs();
            let db = (curr[2] as i32 - prev[2] as i32).abs();
            ((dr << 1) + dg + db) >> 2 // Based on Q. Yang's weight formula
        }
        _ => unreachable!(), // LLVM will prune other branches
    }
}

pub fn normalize<const N_CH_K: usize>(output_and_norm_buf: &[f32], output: &mut [f32]) {
    let n_ch_signal = N_CH_K - 1; // Last channel is normalization factor
    output_and_norm_buf
        .par_chunks_exact(N_CH_K)
        .zip(output.par_chunks_exact_mut(n_ch_signal))
        .for_each(|(unnormed_px, norm_px)| {
            let inv_weight = 1.0 / unnormed_px[n_ch_signal];
            for c in 0..n_ch_signal {
                // Divide by normalization factor and clamp
                norm_px[c] = unnormed_px[c] * inv_weight;
            }
        });
}

pub fn pad_signal_with_weights<const N_CH_K: usize>(
    signal: &[f32],
    output_and_norm_buf: &mut [f32],
    width: usize,
    height: usize,
    val: f32,
) {
    let n_ch_signal = N_CH_K - 1;
    // let mut padded_signal = vec![0.0f32; width * height * N_CH_K];
    for i in 0..width * height {
        let signal_px_idx = i * n_ch_signal;
        let padded_px_idx = i * N_CH_K;
        for c in 0..n_ch_signal {
            output_and_norm_buf[padded_px_idx + c] = signal[signal_px_idx + c];
        }
        output_and_norm_buf[padded_px_idx + n_ch_signal] = val; // Initialize normalization factor to 1
    }
}

pub fn transpose_tiled_hwc<T: Copy + Default, const N_CH: usize>(
    input: &[T],
    output: &mut [T],
    src_h: usize,
    src_w: usize,
) {
    const TILE_SIZE: usize = 16;

    for r_outer in (0..src_h).step_by(TILE_SIZE) {
        for c_outer in (0..src_w).step_by(TILE_SIZE) {
            let r_end = std::cmp::min(r_outer + TILE_SIZE, src_h);
            let c_end = std::cmp::min(c_outer + TILE_SIZE, src_w);

            for row in r_outer..r_end {
                for col in c_outer..c_end {
                    // src is (row, col) in a (src_h, src_w) image
                    let src_idx = (row * src_w + col) * N_CH;
                    // dst is (col, row) in a (src_w, src_h) image
                    let dst_idx = (col * src_h + row) * N_CH;

                    // This is the core "Block Copy"
                    // If c is small (3 or 4), the compiler often inlines this
                    let s = &input[src_idx..src_idx + N_CH];
                    let d = &mut output[dst_idx..dst_idx + N_CH];
                    d.copy_from_slice(s);
                }
            }
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_pad() {
        let (h, w, n_ch, val) = (3, 5, 3, 3_f32);
        let signal = vec![0.0_f32; h * w * n_ch];
        let mut out_buf = vec![0.0_f32; h * w * (n_ch + 1)];

        pad_signal_with_weights::<4>(&signal, &mut out_buf, w, h, val);
        let padded_sum: f32 = out_buf.iter().sum();
        assert!(padded_sum == (h * w) as f32 * val);
    }

    #[test]
    fn test_transpose() {
        let signal: Vec<f32> = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let signal_tr_gt: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

        let mut signal_tr = vec![0.0_f32; 9];
        transpose_tiled_hwc::<f32, 1>(&signal, &mut signal_tr, 3, 3);

        for i in 0..9 {
            let a = signal_tr[i];
            let b = signal_tr_gt[i];
            assert_eq!(a, b)
        }
    }

    #[test]
    fn test_normalize() {
        let signal: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.5];
        let signal_norm_gt: Vec<f32> = vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0];

        let mut signal_norm = vec![0.0_f32; 9];
        normalize::<4>(&signal, &mut signal_norm);
        for i in 0..9 {
            let a = signal_norm[i];
            let b = signal_norm_gt[i];
            assert_eq!(a, b)
        }
    }
    #[test]
    fn test_color_dist() {
        let grey_px_1 = [100u8];
        let grey_px_2 = [150u8];
        let grey_px_3 = [200u8];

        assert_eq!(calculate_dist::<1>(&grey_px_1, &grey_px_2), 50);
        assert_eq!(calculate_dist::<1>(&grey_px_2, &grey_px_3), 50);
        assert_eq!(calculate_dist::<1>(&grey_px_1, &grey_px_3), 100);
    }
}
