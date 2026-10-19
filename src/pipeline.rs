//! MapAnything reconstruction: one forward pass over all images, then multi-view cleaning.

use std::time::Instant;

use anyhow::Result;
use candle_core::Device;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use crate::cleaning::{DepthView, zero_conf_in_front_of_other_views};
use crate::image_input::InputImage;
use crate::model::{MapAnything, ViewOutput};
use crate::point_map::PointMap;
use crate::pose_prior::PoseInputs;
use crate::reconstruction::{Camera, Reconstruction, relative_to_first};

/// Intrinsics are fit on every (side / this)-th pixel along each side, as upstream.
const INTRINSICS_SAMPLES_PER_SIDE: usize = 50;

/// Pinhole intrinsics from the camera-frame points: per axis, the least-squares (center,
/// focal) of pixel = center + focal * (coordinate / z), as upstream's
/// `recover_pinhole_intrinsics_from_ray_directions`.
fn fit_intrinsics(view: &ViewOutput) -> ([f64; 2], [f64; 2]) {
    let step_y = (view.height / INTRINSICS_SAMPLES_PER_SIDE).max(1);
    let step_x = (view.width / INTRINSICS_SAMPLES_PER_SIDE).max(1);
    let mut normal = [Matrix2::<f64>::zeros(); 2];
    let mut rhs = [Vector2::<f64>::zeros(); 2];
    for y in (0..view.height).step_by(step_y) {
        for x in (0..view.width).step_by(step_x) {
            let [px, py, pz] = view.points_in_camera[y * view.width + x].map(|c| c as f64);
            for (axis, (pixel, coordinate)) in
                [(x as f64, px), (y as f64, py)].into_iter().enumerate()
            {
                let ratio = coordinate / pz;
                normal[axis] += Matrix2::new(1.0, ratio, ratio, ratio * ratio);
                rhs[axis] += Vector2::new(pixel, ratio * pixel);
            }
        }
    }
    let solve = |axis: usize| -> Vector2<f64> {
        normal[axis]
            .lu()
            .solve(&rhs[axis])
            .expect("rays must vary across the image")
    };
    let (x, y) = (solve(0), solve(1));
    ([x[0], y[0]], [x[1], y[1]])
}

/// Ratio of the model's translation lengths (relative to view 0) to the given ones, over
/// the views with a given pose: the model's scale in units of the given poses' scale.
fn scale_ratio(relative: &[(Matrix3<f64>, Vector3<f64>)], priors: &PoseInputs) -> f64 {
    let reference = priors.priors[0].expect("view 0 must have a pose");
    let (mut model_sum, mut given_sum) = (0.0, 0.0);
    for (prior, (_, translation)) in priors.priors.iter().zip(relative) {
        if let Some(prior) = prior {
            model_sum += translation.norm();
            given_sum += (prior.translation - reference.translation).norm();
        }
    }
    if given_sum > 0.0 { model_sum / given_sum } else { 1.0 }
}

/// Camera-to-world poses of every view when known poses were given: the known ones, and
/// for views without one the model's prediction expressed relative to view 0, which is
/// known. The world is the frame of the given poses, at the model's scale: the dense
/// points are at that scale, and even with metric poses the model does not adopt the
/// given scale exactly.
fn poses_from_priors(
    predicted: &[(Matrix3<f64>, Vector3<f64>)],
    priors: &PoseInputs,
) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
    let relative = relative_to_first(predicted);
    let reference = priors.priors[0].expect("view 0 must have a pose");
    let scale = scale_ratio(&relative, priors);
    priors
        .priors
        .iter()
        .zip(&relative)
        .map(|(prior, (rotation, translation))| match prior {
            Some(prior) => (
                prior.rotation,
                reference.translation * scale + (prior.translation - reference.translation) * scale,
            ),
            None => (
                reference.rotation * rotation,
                reference.translation * scale + reference.rotation * translation,
            ),
        })
        .collect()
}

/// Prints how far the model's predicted poses are from the given ones (relative to view
/// 0, after scaling), a check that the pose input is understood.
fn report_pose_agreement(predicted: &[(Matrix3<f64>, Vector3<f64>)], priors: &PoseInputs) {
    let relative = relative_to_first(predicted);
    let reference = priors.priors[0].expect("view 0 must have a pose");
    let given: Vec<Option<(Matrix3<f64>, Vector3<f64>)>> = priors
        .priors
        .iter()
        .map(|prior| {
            prior.map(|p| {
                (
                    reference.rotation.transpose() * p.rotation,
                    reference.rotation.transpose() * (p.translation - reference.translation),
                )
            })
        })
        .collect();
    let scale = scale_ratio(&relative, priors);
    let (mut angle_sum, mut dist_sum, mut count) = (0.0, 0.0, 0);
    for (g, (r, t)) in given.iter().zip(&relative) {
        if let Some((gr, gt)) = g {
            let cos = ((gr.transpose() * r).trace() - 1.0) / 2.0;
            angle_sum += cos.clamp(-1.0, 1.0).acos().to_degrees();
            dist_sum += (t - gt * scale).norm() / (gt.norm() * scale).max(1e-9);
            count += 1;
        }
    }
    if count > 0 {
        println!(
            "predicted poses vs given: mean rotation difference {:.2} degrees, mean position \
             difference {:.1}% of the distance to view 0; the model's scale is {scale:.3} x the \
             given one (one pose unit = {:.1} mm for the model)",
            angle_sum / count as f64,
            100.0 * dist_sum / count as f64,
            1000.0 * scale * if priors.metric { priors.unit_m } else { 1.0 }
        );
    }
}

pub fn reconstruct(
    model: &MapAnything,
    images: &[InputImage],
    priors: Option<&PoseInputs>,
    device: &Device,
) -> Result<Reconstruction> {
    let started = Instant::now();
    let tensors: Vec<_> = images.iter().map(|image| image.tensor.clone()).collect();
    let intrinsics: Vec<_> = images.iter().map(|image| image.intrinsics).collect();
    let views = model.forward(&tensors, &intrinsics, priors)?;
    device.synchronize()?;
    println!(
        "ran MapAnything on {} images in {:.2?}",
        images.len(),
        started.elapsed()
    );

    let predicted: Vec<_> = views
        .iter()
        .map(|view| (view.rotation, view.translation))
        .collect();
    let poses = match priors {
        Some(priors) => {
            report_pose_agreement(&predicted, priors);
            poses_from_priors(&predicted, priors)
        }
        None => relative_to_first(&predicted),
    };
    let fitted: Vec<([f64; 2], [f64; 2])> = views.iter().map(fit_intrinsics).collect();
    let mut point_maps: Vec<PointMap> = views
        .iter()
        .zip(&poses)
        .zip(images)
        .map(|((view, (rotation, translation)), image)| PointMap {
            height: view.height,
            width: view.width,
            points: view
                .points_in_camera
                .iter()
                .map(|point| {
                    let world =
                        rotation * Vector3::from_fn(|axis, _| point[axis] as f64) + translation;
                    [world.x as f32, world.y as f32, world.z as f32]
                })
                .collect(),
            // Pixels the model calls ambiguous are rejected, like inconsistent ones.
            conf: view
                .conf
                .iter()
                .zip(&view.unambiguous)
                .map(|(&conf, &unambiguous)| if unambiguous { conf } else { 0.0 })
                .collect(),
            foreground: image.foreground.clone(),
        })
        .collect();

    let depths: Vec<Vec<f32>> = views
        .iter()
        .map(|view| view.points_in_camera.iter().map(|point| point[2]).collect())
        .collect();
    let depth_views: Vec<DepthView> = poses
        .iter()
        .zip(&fitted)
        .zip(&depths)
        .map(
            |(((rotation, translation), (center, focal)), depth)| DepthView {
                rotation: *rotation,
                translation: *translation,
                focal: (focal[0] + focal[1]) / 2.0,
                principal_point: *center,
                depth,
            },
        )
        .collect();
    zero_conf_in_front_of_other_views(&mut point_maps, &depth_views);

    let cameras = poses
        .into_iter()
        .zip(&fitted)
        .map(|((rotation, translation), (_, focal))| Camera {
            rotation,
            translation,
            focal: (focal[0] + focal[1]) / 2.0,
        })
        .collect();
    Ok(Reconstruction {
        point_maps,
        cameras,
    })
}
