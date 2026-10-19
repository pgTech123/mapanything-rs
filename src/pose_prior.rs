//! Known camera poses as model input. MapAnything encodes each view's pose relative to
//! view 0 (quaternion and translation), with translations normalized by their mean length;
//! that length is given to the model only when the poses are metric (in meters), so that
//! the output comes out in the same units.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

/// A camera-to-world pose, OpenCV camera convention (x right, y down, z forward).
#[derive(Clone, Copy, Debug)]
pub struct PosePrior {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

pub struct PoseInputs {
    /// One entry per image; view 0 must have a pose.
    pub priors: Vec<Option<PosePrior>>,
    /// Whether the translations are in meters.
    pub metric: bool,
    /// Meters per unit of the poses as given by the user (1 when not metric); the
    /// translations above are already in meters when metric.
    pub unit_m: f64,
}

/// What the encoders see for one view.
#[derive(Clone, Copy, Debug)]
pub struct PoseEncoding {
    /// Rotation relative to view 0, (x, y, z, w).
    pub quaternion: [f32; 4],
    /// Translation relative to view 0, divided by `scale`.
    pub translation: [f32; 3],
}

/// Unit quaternion (x, y, z, w) with w >= 0.
fn quaternion(rotation: &Matrix3<f64>) -> UnitQuaternion<f64> {
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*rotation));
    if q.w < 0.0 { UnitQuaternion::new_unchecked(-q.into_inner()) } else { q }
}

impl PoseInputs {
    /// Per-view encodings and the normalization factor (mean translation length relative
    /// to view 0 over the views that moved), as upstream's
    /// `transform_pose_using_quats_and_trans_2_to_1` and `normalize_pose_translations`.
    pub fn encodings(&self) -> (Vec<Option<PoseEncoding>>, f64) {
        let reference = self.priors[0].expect("view 0 must have a pose");
        let q0_inv = quaternion(&reference.rotation).inverse();
        let r0_inv = reference.rotation.transpose();
        let relative: Vec<Option<(UnitQuaternion<f64>, Vector3<f64>)>> = self
            .priors
            .iter()
            .map(|prior| {
                prior.map(|p| {
                    (q0_inv * quaternion(&p.rotation), r0_inv * (p.translation - reference.translation))
                })
            })
            .collect();
        let lengths: Vec<f64> = relative
            .iter()
            .flatten()
            .map(|(_, t)| t.norm())
            .filter(|&n| n > 0.0)
            .collect();
        let scale = (lengths.iter().sum::<f64>() / (lengths.len() as f64 + 1e-8)).max(1e-8);
        let encodings = relative
            .into_iter()
            .map(|rel| {
                rel.map(|(q, t)| PoseEncoding {
                    quaternion: [q.i as f32, q.j as f32, q.k as f32, q.w as f32],
                    translation: [(t.x / scale) as f32, (t.y / scale) as f32, (t.z / scale) as f32],
                })
            })
            .collect();
        (encodings, scale)
    }
}
