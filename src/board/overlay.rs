//! Debug overlay per photo: detected corners, the corners placed on the board, and the
//! whole board (also its unseen part) projected with the estimated pose.

use image::{Rgb, RgbImage};

use crate::board::geometry::{Intrinsics, Pose};
use crate::board::{Estimate, ViewResult};

fn put(img: &mut RgbImage, x: i64, y: i64, color: [u8; 3]) {
    if x >= 0 && y >= 0 && x < img.width() as i64 && y < img.height() as i64 {
        img.put_pixel(x as u32, y as u32, Rgb(color));
    }
}

pub fn circle(img: &mut RgbImage, c: [f64; 2], radius: f64, color: [u8; 3]) {
    let steps = (radius * 8.0).max(12.0) as usize;
    for k in 0..steps {
        let t = k as f64 / steps as f64 * std::f64::consts::TAU;
        put(img, (c[0] + radius * t.cos()).round() as i64, (c[1] + radius * t.sin()).round() as i64, color);
    }
}

pub fn disc(img: &mut RgbImage, c: [f64; 2], radius: f64, color: [u8; 3]) {
    let r = radius.ceil() as i64;
    for dy in -r..=r {
        for dx in -r..=r {
            if ((dx * dx + dy * dy) as f64) <= radius * radius {
                put(img, c[0].round() as i64 + dx, c[1].round() as i64 + dy, color);
            }
        }
    }
}

pub fn line(img: &mut RgbImage, a: [f64; 2], b: [f64; 2], color: [u8; 3], thick: bool) {
    let n = (a[0] - b[0]).hypot(a[1] - b[1]).ceil().max(1.0) as usize;
    for k in 0..=n {
        let t = k as f64 / n as f64;
        let (x, y) = ((a[0] + t * (b[0] - a[0])).round() as i64, (a[1] + t * (b[1] - a[1])).round() as i64);
        put(img, x, y, color);
        if thick {
            put(img, x + 1, y, color);
            put(img, x, y + 1, color);
        }
    }
}

const DETECTED: [u8; 3] = [255, 60, 60];
const USED: [u8; 3] = [40, 255, 40];
const EXTRAPOLATED: [u8; 3] = [0, 220, 255];
const OUTLINE: [u8; 3] = [60, 120, 255];
const AXIS_X: [u8; 3] = [255, 0, 0];
const AXIS_Y: [u8; 3] = [0, 255, 0];
const AXIS_Z: [u8; 3] = [60, 60, 255];

fn project(k: &Intrinsics, pose: &Pose, p: nalgebra::Vector3<f64>) -> Option<[f64; 2]> {
    let c = pose.to_camera(p);
    (c[2] > 0.0).then(|| k.project(c))
}

pub fn draw(estimate: &Estimate, view: &ViewResult) -> RgbImage {
    let mut img = view.rgb.clone();
    for c in &view.corners {
        circle(&mut img, c.pos, 4.0, DETECTED);
    }
    let (n_long, n_short) = estimate.board;
    let sq = estimate.square;
    if let Some(pose) = &view.pose {
        let k = &estimate.intrinsics;
        let pt = |x: f64, y: f64, z: f64| project(k, pose, nalgebra::Vector3::new(x * sq, y * sq, z * sq));
        // Outer edge of the board: one square beyond the inner corners.
        let outline = [(-1.0, -1.0), (n_long as f64, -1.0), (n_long as f64, n_short as f64), (-1.0, n_short as f64)];
        for e in 0..4 {
            if let (Some(a), Some(b)) = (pt(outline[e].0, outline[e].1, 0.0), pt(outline[(e + 1) % 4].0, outline[(e + 1) % 4].1, 0.0)) {
                line(&mut img, a, b, OUTLINE, true);
            }
        }
        // Every inner corner, with the estimated warp; hollow where not observed.
        let lattice = view.lattice.as_ref().unwrap();
        for i in 0..n_long {
            for j in 0..n_short {
                let w = estimate.corner_world(i, j);
                let Some(p) = project(k, pose, w) else { continue };
                if let Some(&obs) = lattice.get(&(i, j)) {
                    disc(&mut img, obs, 3.0, USED);
                    line(&mut img, obs, p, USED, false);
                } else {
                    circle(&mut img, p, 3.0, EXTRAPOLATED);
                }
            }
        }
        // Board axes at the origin corner.
        if let Some(o) = pt(0.0, 0.0, 0.0) {
            for (axis, color) in [((2.0, 0.0, 0.0), AXIS_X), ((0.0, 2.0, 0.0), AXIS_Y), ((0.0, 0.0, 2.0), AXIS_Z)] {
                if let Some(e) = pt(axis.0, axis.1, axis.2) {
                    line(&mut img, o, e, color, true);
                }
            }
            circle(&mut img, o, 8.0, [255, 255, 255]);
        }
    } else if let Some(lattice) = &view.raw_lattice {
        // No pose: show the lattice that was found and its grid, so the failure can be
        // understood.
        for (&(i, j), &p) in lattice {
            for (di, dj) in [(1, 0), (0, 1)] {
                if let Some(&q) = lattice.get(&(i + di, j + dj)) {
                    line(&mut img, p, q, [255, 200, 0], false);
                }
            }
            circle(&mut img, p, 5.0, [255, 200, 0]);
        }
    }
    img
}
