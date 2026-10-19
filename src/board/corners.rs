//! Checkerboard X-corner detection: saddle points of the smoothed image, kept when a ring
//! of samples around them shows the two-dark-two-bright pattern, then refined to subpixel
//! precision the way OpenCV's cornerSubPix does (gradients around a corner are orthogonal
//! to the vector from the corner).

use crate::board::photo::GrayImage;

#[derive(Clone, Copy, Debug)]
pub struct Corner {
    pub pos: [f64; 2],
    /// Ring pattern score in [0, 1], 1 for a perfect checker corner.
    pub score: f32,
}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as i64;
    let mut k: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

/// Separable Gaussian blur with edge clamping.
pub fn blur(image: &GrayImage, sigma: f32) -> GrayImage {
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as i64;
    let (w, h) = (image.width, image.height);
    let mut tmp = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let sx = (x as i64 + i as i64 - r).clamp(0, w as i64 - 1) as usize;
                s += kv * image.at(sx, y);
            }
            tmp.data[y * w + x] = s;
        }
    }
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let sy = (y as i64 + i as i64 - r).clamp(0, h as i64 - 1) as usize;
                s += kv * tmp.at(x, sy);
            }
            out.data[y * w + x] = s;
        }
    }
    out
}

/// Central-difference gradients (dx, dy) of `image`.
pub fn gradients(image: &GrayImage) -> (GrayImage, GrayImage) {
    let (w, h) = (image.width, image.height);
    let mut gx = GrayImage::new(w, h);
    let mut gy = GrayImage::new(w, h);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            gx.data[y * w + x] = 0.5 * (image.at(x + 1, y) - image.at(x - 1, y));
            gy.data[y * w + x] = 0.5 * (image.at(x, y + 1) - image.at(x, y - 1));
        }
    }
    (gx, gy)
}

/// Saddle response -det(Hessian) = Ixy^2 - Ixx*Iyy, positive at checker corners.
fn saddle_response(blurred: &GrayImage) -> GrayImage {
    let (w, h) = (blurred.width, blurred.height);
    let mut out = GrayImage::new(w, h);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = blurred.at(x, y);
            let ixx = blurred.at(x + 1, y) - 2.0 * c + blurred.at(x - 1, y);
            let iyy = blurred.at(x, y + 1) - 2.0 * c + blurred.at(x, y - 1);
            let ixy = 0.25
                * (blurred.at(x + 1, y + 1) - blurred.at(x - 1, y + 1) - blurred.at(x + 1, y - 1)
                    + blurred.at(x - 1, y - 1));
            out.data[y * w + x] = ixy * ixy - ixx * iyy;
        }
    }
    out
}

/// Second-harmonic score and phase of the intensity profile on a ring of `radius` around
/// (x, y); None when the ring leaves the image or has no contrast.
fn ring_pattern(image: &GrayImage, x: f64, y: f64, radius: f64) -> Option<(f32, f32)> {
    const N: usize = 32;
    let mut samples = [0f32; N];
    for (k, s) in samples.iter_mut().enumerate() {
        let theta = k as f64 * std::f64::consts::TAU / N as f64;
        *s = image.sample(x + radius * theta.cos(), y + radius * theta.sin())?;
    }
    let mean = samples.iter().sum::<f32>() / N as f32;
    let (mut c2, mut s2, mut var) = (0f32, 0f32, 0f32);
    for (k, s) in samples.iter().enumerate() {
        let v = s - mean;
        let theta = k as f32 * std::f32::consts::TAU / N as f32;
        c2 += v * (2.0 * theta).cos();
        s2 += v * (2.0 * theta).sin();
        var += v * v;
    }
    let rms = (var / N as f32).sqrt();
    if rms < 0.03 {
        return None;
    }
    let amplitude = 2.0 / N as f32 * (c2 * c2 + s2 * s2).sqrt();
    let score = amplitude / (std::f32::consts::SQRT_2 * rms);
    Some((score.min(1.0), s2.atan2(c2)))
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(std::f32::consts::TAU);
    d.min(std::f32::consts::TAU - d)
}

/// Iterative subpixel refinement; returns None if it drifts away from the start.
fn refine_subpixel(gx: &GrayImage, gy: &GrayImage, start: [f64; 2], radius: i64) -> Option<[f64; 2]> {
    let mut c = start;
    let sigma = radius as f64 * 0.6;
    for _ in 0..10 {
        let (mut a, mut b, mut d) = (0.0, 0.0, 0.0);
        let (mut bx, mut by) = (0.0, 0.0);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (px, py) = (c[0] + dx as f64, c[1] + dy as f64);
                let (Some(ix), Some(iy)) = (gx.sample(px, py), gy.sample(px, py)) else {
                    return None;
                };
                let w = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma * sigma)).exp();
                let (ix, iy) = (ix as f64, iy as f64);
                let (gxx, gxy, gyy) = (w * ix * ix, w * ix * iy, w * iy * iy);
                a += gxx;
                b += gxy;
                d += gyy;
                bx += gxx * px + gxy * py;
                by += gxy * px + gyy * py;
            }
        }
        let det = a * d - b * b;
        if det.abs() < 1e-12 {
            return None;
        }
        let next = [(d * bx - b * by) / det, (a * by - b * bx) / det];
        let step = (next[0] - c[0]).hypot(next[1] - c[1]);
        c = next;
        if (c[0] - start[0]).hypot(c[1] - start[1]) > radius as f64 {
            return None;
        }
        if step < 0.005 {
            break;
        }
    }
    Some(c)
}

/// Detects checker corners in `image` (working resolution).
pub fn detect_corners(image: &GrayImage) -> Vec<Corner> {
    let blurred = blur(image, 1.2);
    let response = saddle_response(&blurred);
    let (gx, gy) = gradients(&blurred);
    let (w, h) = (image.width, image.height);
    let max_response = response.data.iter().cloned().fold(0f32, f32::max);
    let threshold = 0.01 * max_response;
    let nms = 3usize;

    let mut candidates = Vec::new();
    for y in nms..h - nms {
        for x in nms..w - nms {
            let r = response.at(x, y);
            if r < threshold {
                continue;
            }
            let mut is_max = true;
            'outer: for dy in 0..=2 * nms {
                for dx in 0..=2 * nms {
                    let v = response.at(x + dx - nms, y + dy - nms);
                    if v > r || (v == r && (dy < nms || (dy == nms && dx < nms))) {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                candidates.push((x as f64, y as f64));
            }
        }
    }

    let mut corners: Vec<Corner> = Vec::new();
    for (x, y) in candidates {
        let Some((s1, phase1)) = ring_pattern(image, x, y, 3.0) else { continue };
        let Some((s2, phase2)) = ring_pattern(image, x, y, 6.5) else { continue };
        if s1 < 0.6 || s2 < 0.6 || angle_diff(phase1, phase2) > 0.6 {
            continue;
        }
        let Some(pos) = refine_subpixel(&gx, &gy, [x, y], 4) else { continue };
        corners.push(Corner { pos, score: s1.min(s2) });
    }
    corners.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Merge detections that refined to the same point.
    let mut kept: Vec<Corner> = Vec::new();
    for c in corners {
        if kept
            .iter()
            .all(|k| (k.pos[0] - c.pos[0]).hypot(k.pos[1] - c.pos[1]) > 3.0)
        {
            kept.push(c);
        }
    }
    kept
}
