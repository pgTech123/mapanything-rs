//! Debug output of the board pose estimation: one overlay per photo, poses.json, and an
//! index.html viewer showing each photo's detection and the cameras in 3D.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::board::{Estimate, overlay, stem};

#[derive(Serialize)]
struct ViewJson {
    name: String,
    path: String,
    overlay: String,
    width: u32,
    height: u32,
    /// Size of the upright photo; intrinsics are for the width x height working image.
    full_width: u32,
    full_height: u32,
    found: bool,
    note: String,
    corners_detected: usize,
    corners_used: usize,
    rms: Option<f64>,
    /// 3x4 camera-to-world (OpenCV camera: x right, y down, z forward), in squares.
    cam_to_world: Option<[[f64; 4]; 3]>,
    center: Option<[f64; 3]>,
}

#[derive(Serialize)]
struct Report {
    board: [i32; 2],
    square: f64,
    intrinsics: crate::board::geometry::Intrinsics,
    board_warp: Vec<f64>,
    rms: f64,
    log: Vec<String>,
    views: Vec<ViewJson>,
}

/// Writes the overlays, `poses.json` and `index.html` into `dir`.
pub fn write_debug(estimate: &Estimate, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut views = Vec::new();
    for view in &estimate.views {
        let overlay_name = format!("{}_board.png", stem(&view.path));
        overlay::draw(estimate, view).save(dir.join(&overlay_name))?;
        views.push(ViewJson {
            name: view.path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
            path: view.path.canonicalize().unwrap_or(view.path.clone()).to_string_lossy().into_owned(),
            overlay: overlay_name,
            width: view.rgb.width(),
            height: view.rgb.height(),
            full_width: view.full_size.0,
            full_height: view.full_size.1,
            found: view.pose.is_some(),
            note: view.note.clone(),
            corners_detected: view.corners.len(),
            corners_used: view.lattice.as_ref().map_or(0, |l| l.len()),
            rms: view.rms,
            cam_to_world: view.pose.as_ref().map(|p| p.cam_to_world()),
            center: view.pose.as_ref().map(|p| {
                let c = p.center();
                [c[0], c[1], c[2]]
            }),
        });
    }
    let report = Report {
        board: [estimate.board.0, estimate.board.1],
        square: estimate.square,
        intrinsics: estimate.intrinsics,
        board_warp: estimate.board_warp.z_offsets.clone(),
        rms: estimate.rms,
        log: estimate.log.clone(),
        views,
    };
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(dir.join("poses.json"), &json)?;
    std::fs::write(dir.join("index.html"), include_str!("viewer.html").replace("/*DATA*/null", &json))?;
    Ok(())
}
