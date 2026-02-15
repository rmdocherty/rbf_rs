use rayon::prelude::*;
use std::f32::consts::E;

const N_COLOURS: usize = 256;

// Inline helper to force specialization of the distance math
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

#[allow(dead_code)]
pub fn recursive_bilateral_filter<'a, const N_CH_SIGNAL: usize, const N_CH_GUIDANCE: usize>(
    signal: &[f32],
    guidance_img: &[u8],
    output_buf: &'a mut [f32],
    width: usize,
    height: usize,
    sigma_spatial: f32,
    sigma_range: f32,
) -> &'a mut [f32] {
    // exponential decaying weight table with falloff defined by $sigma_range
    // i.e, if two pixels are 4 intensity apart, the weight is $colour_weight_range_table[3]
    let inverse_sigma_for_range_table = 1.0 / (sigma_range * N_COLOURS as f32);
    let colour_dist_weight_table: [f32; N_COLOURS] = (0..N_COLOURS)
        .map(|i| E.powf(-(i as f32) * inverse_sigma_for_range_table))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // alpha is spatial weight for filter
    let alpha_h = E.powf(-f32::sqrt(2.0) / (sigma_spatial * (width as f32).max(height as f32)));

    rbf_horizontal_parallel::<N_CH_SIGNAL, N_CH_GUIDANCE>(
        signal,
        guidance_img,
        output_buf,
        width,
        alpha_h,
        &colour_dist_weight_table,
    );

    output_buf
}

pub fn rbf_horizontal_parallel<const N_CH_SIGNAL: usize, const N_CH_GUIDANCE: usize>(
    signal: &[f32],
    guidance_img: &[u8],
    output_and_norm_buf: &mut [f32],
    width: usize,
    alpha_h: f32,
    colour_dist_weight_table: &[f32; N_COLOURS],
) {
    let inv_alpha_h = 1.0 - alpha_h;
    let signal_row_size = width * N_CH_SIGNAL;
    let guidance_row_size = width * N_CH_GUIDANCE;

    output_and_norm_buf
        .par_chunks_exact_mut(signal_row_size)
        .zip(signal.par_chunks_exact(signal_row_size))
        .zip(guidance_img.par_chunks_exact(guidance_row_size))
        .for_each(
            |((output_and_norm_buf_row, signal_row), guidance_img_row)| {
                // =========== Forward Pass (Left-to-Right) ===========
                let mut prev_guidance_px = &guidance_img_row[0..N_CH_GUIDANCE];
                // we use '10' rather than N_CH_SIGNAL as stable rust can't guarantee size of const generics yet
                let mut prev_filtered_px = [0.0f32; 11];
                // Initialize first pixel
                for c in 0..N_CH_SIGNAL {
                    let val = signal_row[c] as f32;
                    output_and_norm_buf_row[c] = val;
                    prev_filtered_px[c] = val;
                }
                // for each x-pixel in the row
                for x in 1..width {
                    let signal_px_idx = x * N_CH_SIGNAL;
                    let guidance_px_idx = x * N_CH_GUIDANCE;

                    let current_signal_px =
                        &signal_row[signal_px_idx..(signal_px_idx + N_CH_SIGNAL)];
                    let current_guidance_px =
                        &guidance_img_row[guidance_px_idx..(guidance_px_idx + N_CH_GUIDANCE)];

                    let colour_dist =
                        calculate_dist::<N_CH_GUIDANCE>(current_guidance_px, prev_guidance_px);
                    let spatial_and_colour_weight =
                        colour_dist_weight_table[colour_dist as usize] * alpha_h;

                    for c in 0..N_CH_SIGNAL {
                        let filtered_val = inv_alpha_h * (current_signal_px[c])
                            + spatial_and_colour_weight * prev_filtered_px[c];
                        output_and_norm_buf_row[signal_px_idx + c] = filtered_val;
                        prev_filtered_px[c] = filtered_val;
                    }

                    // Filter normalization value (the last channel) separately, as doesn't have corresponding guidance value
                    let norm_val_idx = signal_px_idx + N_CH_SIGNAL;
                    let filtered_norm_val = inv_alpha_h * 1.0
                        + spatial_and_colour_weight * prev_filtered_px[N_CH_SIGNAL];
                    output_and_norm_buf_row[norm_val_idx] = filtered_norm_val;
                    prev_filtered_px[N_CH_SIGNAL] = filtered_norm_val;

                    prev_guidance_px = current_guidance_px;
                }

                // =========== Backward Pass (Right-to-Left) ===========
                let mut prev_guidance_px_rev =
                    &guidance_img_row[(width - 1) * N_CH_GUIDANCE..width * N_CH_GUIDANCE];
                let mut prev_filtered_px_rev = [0.0f32; 10];

                // Initialize with the last pixel of the signal
                for c in 0..N_CH_SIGNAL {
                    let val = signal_row[(width - 1) * N_CH_SIGNAL + c] as f32;
                    prev_filtered_px_rev[c] = val;
                }

                for x in (0..width - 1).rev() {
                    let signal_px_idx = x * N_CH_SIGNAL;
                    let guidance_px_idx = x * N_CH_GUIDANCE;

                    let current_signal_px =
                        &signal_row[signal_px_idx..(signal_px_idx + N_CH_SIGNAL)];
                    let current_guidance_px =
                        &guidance_img_row[guidance_px_idx..(guidance_px_idx + N_CH_GUIDANCE)];

                    let colour_dist =
                        calculate_dist::<N_CH_GUIDANCE>(current_guidance_px, prev_guidance_px_rev);
                    let spatial_and_colour_weight =
                        colour_dist_weight_table[colour_dist as usize] * alpha_h;

                    for c in 0..N_CH_SIGNAL {
                        let filtered_val_rev = inv_alpha_h * (current_signal_px[c])
                            + spatial_and_colour_weight * prev_filtered_px[c];
                        // average the reverse pass ($filtered_val) with the forward pass ``
                        let filtered_val_fwd = output_and_norm_buf_row[signal_px_idx + c];
                        output_and_norm_buf_row[signal_px_idx + c] =
                            0.5 * (filtered_val_rev + filtered_val_fwd);
                        prev_filtered_px[c] = filtered_val_rev;
                    }

                    let norm_val_idx = signal_px_idx + N_CH_SIGNAL;
                    let filtered_norm_val = inv_alpha_h * 1.0
                        + spatial_and_colour_weight * prev_filtered_px_rev[N_CH_SIGNAL];
                    let norm_val_fwd = output_and_norm_buf_row[norm_val_idx];
                    output_and_norm_buf_row[norm_val_idx] =
                        0.5 * (filtered_norm_val + norm_val_fwd);
                    prev_filtered_px[N_CH_SIGNAL] = filtered_norm_val;

                    prev_guidance_px_rev = current_guidance_px;
                }
            },
        );
}

const TILE_SIZE: usize = 32; // Optimized for L1/L2 cache line sizes

pub fn transpose_tiled<const N_CH_TOTAL: usize>(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
) {
    // We process the image in TILE_SIZE x TILE_SIZE blocks of pixels
    let num_tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let num_tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

    // Parallelize across the Y-tiles
    (0..num_tiles_y).into_par_iter().for_each(|ty| {
        let y_start = ty * TILE_SIZE;
        let y_end = (y_start + TILE_SIZE).min(height);

        for tx in 0..num_tiles_x {
            let x_start = tx * TILE_SIZE;
            let x_end = (x_start + TILE_SIZE).min(width);

            // Transpose the individual pixels within this tile
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let in_idx = (y * width + x) * N_CH_TOTAL;
                    let out_idx = (x * height + y) * N_CH_TOTAL;

                    // Move the entire pixel (Signal + Norm) at once
                    // LLVM can often vectorize this loop if N_CH_TOTAL is a constant
                    let src = &input[in_idx..in_idx + N_CH_TOTAL];
                    let dst = &mut output[out_idx..out_idx + N_CH_TOTAL];
                    dst.copy_from_slice(src);
                }
            }
        }
    });
}
