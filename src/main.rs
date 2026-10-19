use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};

use mapanything_rs::board;
use mapanything_rs::model::MapAnything;
use mapanything_rs::pipeline::reconstruct;
use mapanything_rs::ply::{confident_points, write_ply};
use mapanything_rs::point_map::PointMap;
use mapanything_rs::pose_prior::{PoseInputs, PosePrior};
use mapanything_rs::preprocess::load_images;

/// MapAnything, Apache 2.0 weights.
const WEIGHTS_REPO: &str = "facebook/map-anything-apache";
const WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
    Bf16,
}

impl From<Precision> for DType {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        }
    }
}

/// Reconstruct a colored 3D point cloud from two or more images of the same scene, with
/// MapAnything: all images in one forward pass.
#[derive(Parser)]
struct Args {
    /// Two or more JPEG or PNG images, view 0 first. Transparent pixels of PNGs are
    /// background: cut the background out for turntable captures (fixed camera, rotating
    /// object).
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Output point cloud (y up), in the camera frame of the first image; in the board frame
    /// (z up) with `--checkerboard`.
    #[arg(long, default_value = "reconstruction.ply")]
    out: PathBuf,
    /// Keep points with confidence >= this. Confidence is 1 + exp(raw), so always > 1, or 0
    /// for points rejected as ambiguous or inconsistent with other views.
    #[arg(long, default_value_t = 1.0)]
    min_conf: f32,
    /// 35 mm-equivalent focal length of the camera, in mm, for every image; overrides EXIF.
    /// Known focal lengths are given to the model instead of letting it guess, which it does
    /// poorly on object-centric captures. Video frames have no EXIF: iPhone main cameras are
    /// about 24-26.
    #[arg(long)]
    focal_35mm: Option<f64>,
    #[arg(long, value_enum, default_value = "f16")]
    precision: Precision,
    /// MapAnything safetensors checkpoint; downloaded from Hugging Face when omitted.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Also save the dense point maps (image{i}.pts3d [H, W, 3], image{i}.conf [H, W],
    /// image{i}.foreground [H, W] u8) and cameras (image{i}.cam_to_world [3, 4],
    /// image{i}.focal in pixels) to this safetensors file, OpenCV camera convention.
    #[arg(long)]
    raw_out: Option<PathBuf>,
    /// Estimate every camera's pose from a checkerboard the object stands on, and give the
    /// poses to the model. The output is then in the board frame (z up, unit: one square)
    /// instead of the first camera's. Photos where the board is not placed uniquely take
    /// part without a pose; if no photo has one, the run continues without poses.
    #[arg(long)]
    checkerboard: bool,
    /// Inner corner counts of the board, long axis first (9x6 for a 10x7 squares board).
    /// Detected from photos showing the whole board when omitted.
    #[arg(long, requires = "checkerboard")]
    board_size: Option<String>,
    /// Side of a board square in meters, to give the poses to the model as metric. Without
    /// it the model chooses the scale. A wrong value does harm; check the printed ratio.
    #[arg(long, requires = "checkerboard")]
    square_m: Option<f64>,
    /// Write the board detection debug output here: an overlay per photo, poses.json and an
    /// index.html viewer with the cameras in 3D.
    #[arg(long, requires = "checkerboard")]
    board_debug: Option<PathBuf>,
    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,
}

/// Diagonal of a 36 x 24 mm frame, the reference of 35 mm-equivalent focal lengths.
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615;

/// Runs the checkerboard pose estimation. Returns the pose inputs (None per image where the
/// board was not placed) and the calibrated focal length as a 35 mm equivalent, or None
/// with a warning when the board is unusable.
fn poses_from_checkerboard(args: &Args) -> Option<(PoseInputs, f64)> {
    let board = match &args.board_size {
        Some(s) => match s.split_once('x').and_then(|(a, b)| Some((a.trim().parse::<i32>().ok()?, b.trim().parse::<i32>().ok()?))) {
            Some((a, b)) => Some((a.max(b), a.min(b))),
            None => {
                eprintln!("warning: --board-size must be like 9x6; detecting the board size instead");
                None
            }
        },
        None => None,
    };
    let options = board::Options { board, focal_35mm: args.focal_35mm, ..board::Options::default() };
    let started = Instant::now();
    let estimate = match board::estimate_poses(&args.images, &options) {
        Ok(estimate) => estimate,
        Err(error) => {
            eprintln!("warning: checkerboard poses unavailable ({error:#}); continuing without poses");
            return None;
        }
    };
    for line in &estimate.log {
        println!("{line}");
    }
    for view in &estimate.views {
        let name = view.path.file_name().unwrap_or_default().to_string_lossy();
        match (&view.pose, view.rms) {
            (Some(pose), Some(rms)) => {
                let c = pose.center();
                println!("  {name}: camera at ({:.2}, {:.2}, {:.2}) squares, {} corners, RMS {rms:.2} px", c[0], c[1], c[2], view.lattice.as_ref().map_or(0, |l| l.len()));
            }
            _ => println!("  {name}: no pose, {}", view.note),
        }
    }
    if let Some(dir) = &args.board_debug {
        match board::report::write_debug(&estimate, dir) {
            Ok(()) => println!("board debug output in {}", dir.join("index.html").display()),
            Err(error) => eprintln!("warning: could not write the board debug output: {error:#}"),
        }
    }
    let found = estimate.views.iter().filter(|v| v.pose.is_some()).count();
    println!("checkerboard poses for {found} of {} photos in {:.1?}", estimate.views.len(), started.elapsed());
    if found == 0 {
        eprintln!("warning: the checkerboard was not placed in any photo; continuing without poses");
        return None;
    }
    let unit_m = args.square_m.unwrap_or(1.0);
    let priors = estimate
        .views
        .iter()
        .map(|view| {
            view.pose.as_ref().map(|pose| {
                let m = pose.cam_to_world();
                PosePrior {
                    rotation: nalgebra::Matrix3::from_fn(|r, c| m[r][c]),
                    translation: nalgebra::Vector3::new(m[0][3], m[1][3], m[2][3]) * unit_m,
                }
            })
        })
        .collect();
    let view = &estimate.views[0];
    let focal_full = estimate.intrinsics.f * view.full_size.0 as f64 / view.rgb.width() as f64;
    let focal_35mm = focal_full * FULL_FRAME_DIAGONAL_MM / (view.full_size.0 as f64).hypot(view.full_size.1 as f64);
    Some((PoseInputs { priors, metric: args.square_m.is_some(), unit_m }, focal_35mm))
}

fn point_map_tensors(index: usize, point_map: &PointMap) -> Result<[(String, Tensor); 3]> {
    let (height, width) = (point_map.height, point_map.width);
    let points: Vec<f32> = point_map.points.iter().flatten().copied().collect();
    let foreground: Vec<u8> = point_map.foreground.iter().map(|&f| f as u8).collect();
    Ok([
        (
            format!("image{index}.pts3d"),
            Tensor::from_vec(points, (height, width, 3), &Device::Cpu)?,
        ),
        (
            format!("image{index}.conf"),
            Tensor::from_slice(&point_map.conf, (height, width), &Device::Cpu)?,
        ),
        (
            format!("image{index}.foreground"),
            Tensor::from_vec(foreground, (height, width), &Device::Cpu)?,
        ),
    ])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    let dtype = DType::from(args.precision);

    let weights = match &args.weights {
        Some(path) => path.clone(),
        None => hf_hub::api::sync::Api::new()?
            .model(WEIGHTS_REPO.to_string())
            .get(WEIGHTS_FILE)?,
    };
    let started = Instant::now();
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device)? };
    let model = MapAnything::new(vb)?;
    println!("loaded {} in {:.2?}", weights.display(), started.elapsed());

    let mut image_paths = args.images.clone();
    let mut focal_35mm = args.focal_35mm;
    let mut poses = None;
    if args.checkerboard
        && let Some((mut inputs, calibrated_focal)) = poses_from_checkerboard(&args)
    {
        // The reference view must have a pose: move the first image with one to the front.
        let first = inputs.priors.iter().position(Option::is_some).expect("at least one pose");
        if first != 0 {
            println!(
                "{} is the reference view: it is the first image with a pose",
                image_paths[first].display()
            );
            image_paths.swap(0, first);
            inputs.priors.swap(0, first);
        }
        let known = inputs.priors.iter().flatten().count();
        println!(
            "camera poses known for {known} of {} images{}; calibrated focal length {calibrated_focal:.1} mm (35 mm equivalent)",
            image_paths.len(),
            if inputs.metric { ", metric" } else { ", scale left to the model" }
        );
        focal_35mm = focal_35mm.or(Some(calibrated_focal));
        poses = Some(inputs);
    }
    let images = load_images(&image_paths, focal_35mm, dtype, &device)?;
    let known = images
        .iter()
        .filter(|image| image.intrinsics.is_some())
        .count();
    println!(
        "camera intrinsics known for {known} of {} images",
        images.len()
    );
    let reconstruction = reconstruct(&model, &images, poses.as_ref(), &device)?;

    if let Some(raw_out) = &args.raw_out {
        let mut tensors: HashMap<String, Tensor> = reconstruction
            .point_maps
            .iter()
            .enumerate()
            .map(|(index, point_map)| point_map_tensors(index, point_map))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        for (index, camera) in reconstruction.cameras.iter().enumerate() {
            let cam_to_world: Vec<f32> = (0..3)
                .flat_map(|row| {
                    [
                        camera.rotation[(row, 0)],
                        camera.rotation[(row, 1)],
                        camera.rotation[(row, 2)],
                        camera.translation[row],
                    ]
                })
                .map(|value| value as f32)
                .collect();
            tensors.insert(
                format!("image{index}.cam_to_world"),
                Tensor::from_vec(cam_to_world, (3, 4), &Device::Cpu)?,
            );
            tensors.insert(
                format!("image{index}.focal"),
                Tensor::new(camera.focal as f32, &Device::Cpu)?,
            );
        }
        candle_core::safetensors::save(&tensors, raw_out)?;
        println!("wrote {}", raw_out.display());
    }

    let mut points = Vec::new();
    for (point_map, image) in reconstruction.point_maps.iter().zip(&images) {
        points.extend(confident_points(point_map, &image.rgb, args.min_conf, poses.is_none())?);
    }
    write_ply(&args.out, &points)?;
    println!("wrote {} points to {}", points.len(), args.out.display());
    Ok(())
}
