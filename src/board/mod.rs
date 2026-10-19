//! Camera pose of each photo from the visible part of a checkerboard the object stands on.
//!
//! [`estimate_poses`] detects the board's inner corners in every photo, places the visible
//! lattice on the physical board (also when the board is partly hidden or cut off, as long
//! as enough of its edges are seen to make the placement unique), calibrates the camera from
//! all photos together, and returns every camera's pose in the board frame: x along the
//! long axis, y along the short axis, z up, in units of one square.

pub mod corners;
pub mod geometry;
pub mod lattice;
pub mod overlay;
pub mod photo;
pub mod report;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use nalgebra::Vector3;
use rayon::prelude::*;

use corners::Corner;
use geometry::{BaOptions, Board, Intrinsics, Observation, Pose, bundle_adjust, focal_from_homographies, homography_dlt, pose_from_homography, reprojection_errors};
use lattice::{BoardSize, Lattice};

#[derive(Clone, Debug)]
pub struct Options {
    /// Integer factor the photos are shrunk by before detection; pixels in the result are
    /// at this resolution.
    pub downscale: u32,
    /// Inner corner counts (long, short); detected from photos showing the whole board
    /// when None.
    pub board: Option<BoardSize>,
    /// Side of a square, the unit of the poses.
    pub square: f64,
    /// 35 mm-equivalent focal length overriding EXIF, as the calibration's start.
    pub focal_35mm: Option<f64>,
    /// Keep the EXIF focal length and the image center instead of calibrating them.
    pub trust_exif: bool,
    /// Assume a perfectly flat board instead of estimating its warp.
    pub flat_board: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { downscale: 4, board: None, square: 1.0, focal_35mm: None, trust_exif: false, flat_board: false }
    }
}

pub struct ViewResult {
    pub path: PathBuf,
    pub rgb: image::RgbImage,
    /// Size of the upright photo before downscaling.
    pub full_size: (u32, u32),
    pub corners: Vec<Corner>,
    /// The lattice as found, with an arbitrary origin and orientation.
    pub raw_lattice: Option<Lattice>,
    /// The lattice placed on the board, when the placement was unique.
    pub lattice: Option<Lattice>,
    /// Why there is no pose, or how the placement was decided.
    pub note: String,
    pub pose: Option<Pose>,
    /// RMS reprojection error of this view's corners, in pixels.
    pub rms: Option<f64>,
}

pub struct Estimate {
    pub board: BoardSize,
    pub square: f64,
    pub intrinsics: Intrinsics,
    pub board_warp: Board,
    pub views: Vec<ViewResult>,
    /// RMS reprojection error over all views, in pixels.
    pub rms: f64,
    pub log: Vec<String>,
}

impl Estimate {
    /// World position of inner corner (i, j), including the estimated warp.
    pub fn corner_world(&self, i: i32, j: i32) -> Vector3<f64> {
        let z = self.board_warp.z_offsets[(i * self.board.1 + j) as usize];
        Vector3::new(i as f64 * self.square, j as f64 * self.square, z)
    }
}

pub fn estimate_poses(paths: &[PathBuf], opts: &Options) -> Result<Estimate> {
    let mut log = Vec::new();
    let photos: Vec<photo::Photo> = paths
        .par_iter()
        .map(|p| photo::load_photo(p, opts.downscale, opts.focal_35mm))
        .collect::<Result<_>>()?;
    let (width, height) = (photos[0].gray.width, photos[0].gray.height);
    if photos.iter().any(|p| p.gray.width != width || p.gray.height != height) {
        bail!("all photos must have the same size");
    }

    let detections: Vec<(Vec<Corner>, Option<Lattice>)> = photos
        .par_iter()
        .map(|p| {
            let c = corners::detect_corners(&p.gray);
            let l = lattice::find_lattice(&c, &p.gray);
            (c, l)
        })
        .collect();

    let board = match opts.board {
        Some(b) => b,
        None => {
            // The most common extent among the lattices, assuming some photos show the
            // whole board.
            let mut counts: HashMap<BoardSize, usize> = HashMap::new();
            for (_, l) in detections.iter().filter_map(|(c, l)| l.as_ref().map(|l| (c, l))) {
                let (a, b) = lattice::extents(l);
                *counts.entry((a.max(b), a.min(b))).or_insert(0) += 1;
            }
            let (&size, &n) = counts
                .iter()
                .max_by_key(|&(size, n)| (*n, size.0 * size.1))
                .context("no lattice found in any photo")?;
            log.push(format!("board inner corners: {}x{} (full extent seen in {n} of {} photos); pass --board-size if wrong", size.0, size.1, photos.len()));
            size
        }
    };

    let mut views: Vec<ViewResult> = Vec::new();
    for (p, (c, l)) in photos.iter().zip(detections) {
        let (lattice, note) = match &l {
            Some(l) => match lattice::place_on_board(l, board, &p.gray) {
                Ok(placement) => (Some(placement.lattice), placement.note),
                Err(e) => (None, e),
            },
            None => (None, format!("no lattice among {} corners", c.len())),
        };
        views.push(ViewResult { path: p.path.clone(), rgb: p.rgb.clone(), full_size: p.full_size, corners: c, raw_lattice: l, lattice, note, pose: None, rms: None });
    }
    let used: Vec<usize> = (0..views.len()).filter(|&i| views[i].lattice.as_ref().is_some_and(|l| l.len() >= 6)).collect();
    if used.len() < 2 {
        bail!("fewer than two photos with a placed board");
    }

    // Calibration.
    let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
    let observations: Vec<Vec<Observation>> = used
        .iter()
        .map(|&i| {
            views[i]
                .lattice
                .as_ref()
                .unwrap()
                .iter()
                .map(|(&(a, b), &px)| Observation {
                    corner: (a * board.1 + b) as usize,
                    world: Vector3::new(a as f64 * opts.square, b as f64 * opts.square, 0.0),
                    pixel: px,
                })
                .collect()
        })
        .collect();
    let homographies: Vec<_> = observations
        .iter()
        .map(|obs| {
            let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = obs.iter().map(|o| ([o.world[0], o.world[1]], o.pixel)).unzip();
            homography_dlt(&src, &dst).unwrap()
        })
        .collect();
    let exif_focal = used.iter().find_map(|&i| photos[i].focal_px);
    let zhang_focal = focal_from_homographies(&homographies, cx, cy);
    let f0 = match (opts.trust_exif, exif_focal, zhang_focal) {
        (false, _, Some(f)) if exif_focal.is_none_or(|e| (f / e - 1.0).abs() < 0.4) => f,
        (_, Some(e), _) => e,
        (_, None, Some(f)) => f,
        _ => bail!("no focal length: pass one with focal_35mm"),
    };
    log.push(format!(
        "focal length: EXIF {} px, from the board {} px, starting at {f0:.1} px",
        exif_focal.map_or("none".into(), |f| format!("{f:.1}")),
        zhang_focal.map_or("none".into(), |f| format!("{f:.1}"))
    ));
    let k = Intrinsics { f: f0, cx, cy, k1: 0.0, k2: 0.0 };
    let poses: Vec<Pose> = homographies
        .iter()
        .map(|h| pose_from_homography(h, &k.matrix()).context("degenerate homography"))
        .collect::<Result<_>>()?;
    let flat = Board { z_offsets: vec![0.0; (board.0 * board.1) as usize] };
    let stage1 = BaOptions { principal_point: false, distortion: false, board_deformation: false, iterations: 20 };
    let (k1, p1, _, rms1) = bundle_adjust(k, &poses, &flat, &observations, stage1);
    let errors = reprojection_errors(&k1, &p1, &flat, &observations);
    let threshold = (4.0 * rms1).max(1.5);
    let mut rejected = 0;
    let observations: Vec<Vec<Observation>> = observations
        .into_iter()
        .zip(&errors)
        .map(|(obs, errs)| {
            obs.into_iter()
                .zip(errs)
                .filter(|&(_, &e)| {
                    if e > threshold {
                        rejected += 1;
                    }
                    e <= threshold
                })
                .map(|(o, _)| o)
                .collect()
        })
        .collect();
    let stage2 = BaOptions {
        principal_point: !opts.trust_exif,
        distortion: true,
        board_deformation: !opts.flat_board,
        iterations: 60,
    };
    let (k2, p2, board_warp, rms2) = bundle_adjust(k1, &p1, &flat, &observations, stage2);
    let max_warp = board_warp.z_offsets.iter().fold(0f64, |m, z| m.max(z.abs()));
    let n_obs: usize = observations.iter().map(|o| o.len()).sum();
    log.push(format!(
        "calibrated: f {:.1} px, center ({:.1}, {:.1}), k1 {:.4}, k2 {:.4}; reprojection RMS {rms1:.2} -> {rms2:.2} px \
         over {n_obs} corners ({rejected} rejected); board warps up to {max_warp:.3} squares out of plane",
        k2.f, k2.cx, k2.cy, k2.k1, k2.k2
    ));
    let errors = reprojection_errors(&k2, &p2, &board_warp, &observations);
    for ((&i, pose), errs) in used.iter().zip(&p2).zip(&errors) {
        views[i].pose = Some(*pose);
        views[i].rms = Some((errs.iter().map(|e| e * e).sum::<f64>() / errs.len().max(1) as f64).sqrt());
    }
    Ok(Estimate { board, square: opts.square, intrinsics: k2, board_warp, views, rms: rms2, log })
}

/// File stem of a photo, for naming its outputs.
pub fn stem(path: &Path) -> String {
    path.file_stem().unwrap_or_default().to_string_lossy().into_owned()
}
