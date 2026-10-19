//! Projective geometry: homographies, Zhang-style focal length from board homographies,
//! pose from a homography, and bundle adjustment of intrinsics, poses and board warp.

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

pub fn apply_homography(h: &Matrix3<f64>, p: [f64; 2]) -> [f64; 2] {
    let v = h * Vector3::new(p[0], p[1], 1.0);
    [v[0] / v[2], v[1] / v[2]]
}

/// Normalizing similarity: centroid to origin, mean distance sqrt(2).
fn normalizer(points: &[[f64; 2]]) -> Matrix3<f64> {
    let n = points.len() as f64;
    let cx = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = points.iter().map(|p| p[1]).sum::<f64>() / n;
    let mean_dist = points.iter().map(|p| (p[0] - cx).hypot(p[1] - cy)).sum::<f64>() / n;
    let s = if mean_dist > 0.0 { 2f64.sqrt() / mean_dist } else { 1.0 };
    Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
}

/// Direct linear transform homography mapping src -> dst (at least 4 points).
pub fn homography_dlt(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }
    let ts = normalizer(src);
    let td = normalizer(dst);
    let mut a = DMatrix::<f64>::zeros(2 * src.len(), 9);
    for (k, (s, d)) in src.iter().zip(dst).enumerate() {
        let s = apply_homography(&ts, *s);
        let d = apply_homography(&td, *d);
        let (x, y, u, v) = (s[0], s[1], d[0], d[1]);
        let row0 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let row1 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for c in 0..9 {
            a[(2 * k, c)] = row0[c];
            a[(2 * k + 1, c)] = row1[c];
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let hvec = vt.row(8);
    let hn = Matrix3::from_iterator(hvec.iter().cloned()).transpose();
    let h = td.try_inverse()? * hn * ts;
    if h[(2, 2)].abs() < 1e-12 {
        return None;
    }
    Some(h / h[(2, 2)])
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Intrinsics {
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
}

impl Intrinsics {
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.f, 0.0, self.cx, 0.0, self.f, self.cy, 0.0, 0.0, 1.0)
    }

    /// Projects a point in camera coordinates (z forward) to a pixel, with radial distortion.
    pub fn project(&self, p: Vector3<f64>) -> [f64; 2] {
        let (x, y) = (p[0] / p[2], p[1] / p[2]);
        let r2 = x * x + y * y;
        let d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        [self.f * x * d + self.cx, self.f * y * d + self.cy]
    }
}

/// Focal length from board homographies, with a known principal point (Zhang's method
/// reduced to one unknown: the two orthonormality constraints per view are linear in
/// 1/f^2).
pub fn focal_from_homographies(hs: &[Matrix3<f64>], cx: f64, cy: f64) -> Option<f64> {
    let t_inv = Matrix3::new(1.0, 0.0, -cx, 0.0, 1.0, -cy, 0.0, 0.0, 1.0);
    let (mut num, mut den) = (0.0, 0.0);
    for h in hs {
        let h = t_inv * h;
        let (h1, h2) = (h.column(0), h.column(1));
        let a = h1[0] * h2[0] + h1[1] * h2[1];
        let b = h1[2] * h2[2];
        num += a * b;
        den += a * a;
        let a = h1[0] * h1[0] + h1[1] * h1[1] - h2[0] * h2[0] - h2[1] * h2[1];
        let b = h1[2] * h1[2] - h2[2] * h2[2];
        num += a * b;
        den += a * a;
    }
    let inv_f2 = -num / den;
    (inv_f2 > 0.0).then(|| inv_f2.sqrt().recip())
}

/// Rigid camera pose: x_cam = r * x_world + t.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
}

impl Pose {
    pub fn to_camera(&self, p: Vector3<f64>) -> Vector3<f64> {
        self.r * p + self.t
    }
    pub fn center(&self) -> Vector3<f64> {
        self.r.inverse() * (-self.t)
    }
    /// 3x4 camera-to-world matrix (rotation columns are the camera axes in world
    /// coordinates, last column the camera center), OpenCV convention.
    pub fn cam_to_world(&self) -> [[f64; 4]; 3] {
        let rt = self.r.inverse().matrix().clone_owned();
        let c = self.center();
        let mut m = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = rt[(i, j)];
            }
            m[i][3] = c[i];
        }
        m
    }
}

/// Pose of a plane z=0 from its homography (board -> pixels) and intrinsics.
pub fn pose_from_homography(h: &Matrix3<f64>, k: &Matrix3<f64>) -> Option<Pose> {
    let m = k.try_inverse()? * h;
    let lambda = 1.0 / m.column(0).norm();
    let mut r1 = m.column(0) * lambda;
    let mut r2 = m.column(1) * lambda;
    let mut t = m.column(2) * lambda;
    if t[2] < 0.0 {
        r1 = -r1;
        r2 = -r2;
        t = -t;
    }
    let r3 = r1.cross(&r2);
    let r = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = r.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        let mut u = u;
        u.column_mut(2).neg_mut();
        r = u * vt;
    }
    Some(Pose { r: Rotation3::from_matrix_unchecked(r), t: Vector3::new(t[0], t[1], t[2]) })
}

/// One observed board corner: which corner, its nominal world point on z=0, and its pixel.
pub struct Observation {
    pub corner: usize,
    pub world: Vector3<f64>,
    pub pixel: [f64; 2],
}

/// The calibration target: an estimated out-of-plane offset per corner (a paper board is
/// never quite flat).
#[derive(Clone, Debug)]
pub struct Board {
    pub z_offsets: Vec<f64>,
}

#[derive(Clone, Copy)]
pub struct BaOptions {
    pub principal_point: bool,
    pub distortion: bool,
    pub board_deformation: bool,
    pub iterations: usize,
}

/// Weight (pixels per unit of offset) pulling board offsets toward zero, which fixes the
/// gauge (a planar tilt of the offsets is indistinguishable from rotating the board frame).
const BOARD_PRIOR_WEIGHT: f64 = 1.0;

fn pack(k: &Intrinsics, poses: &[Pose], board: &Board) -> DVector<f64> {
    let mut v = Vec::with_capacity(5 + 6 * poses.len() + board.z_offsets.len());
    v.extend_from_slice(&[k.f, k.cx, k.cy, k.k1, k.k2]);
    for p in poses {
        let w = p.r.scaled_axis();
        v.extend_from_slice(&[w[0], w[1], w[2], p.t[0], p.t[1], p.t[2]]);
    }
    v.extend_from_slice(&board.z_offsets);
    DVector::from_vec(v)
}

fn unpack(v: &DVector<f64>, n: usize) -> (Intrinsics, Vec<Pose>, Board) {
    let k = Intrinsics { f: v[0], cx: v[1], cy: v[2], k1: v[3], k2: v[4] };
    let poses = (0..n)
        .map(|i| {
            let o = 5 + 6 * i;
            Pose {
                r: Rotation3::from_scaled_axis(Vector3::new(v[o], v[o + 1], v[o + 2])),
                t: Vector3::new(v[o + 3], v[o + 4], v[o + 5]),
            }
        })
        .collect();
    let board = Board { z_offsets: v.as_slice()[5 + 6 * n..].to_vec() };
    (k, poses, board)
}

fn residuals(v: &DVector<f64>, views: &[Vec<Observation>]) -> DVector<f64> {
    let (k, poses, board) = unpack(v, views.len());
    let n: usize = views.iter().map(|o| o.len()).sum();
    let mut r = DVector::zeros(2 * n + board.z_offsets.len());
    let mut idx = 0;
    for (obs, pose) in views.iter().zip(&poses) {
        for o in obs {
            let world = o.world + Vector3::new(0.0, 0.0, board.z_offsets[o.corner]);
            let p = k.project(pose.to_camera(world));
            r[idx] = p[0] - o.pixel[0];
            r[idx + 1] = p[1] - o.pixel[1];
            idx += 2;
        }
    }
    for z in &board.z_offsets {
        r[idx] = BOARD_PRIOR_WEIGHT * z;
        idx += 1;
    }
    r
}

/// Levenberg-Marquardt refinement of intrinsics, poses and board deformation on
/// reprojection error. Returns the refined values and the RMS reprojection error in pixels.
pub fn bundle_adjust(
    k: Intrinsics,
    poses: &[Pose],
    board: &Board,
    views: &[Vec<Observation>],
    opts: BaOptions,
) -> (Intrinsics, Vec<Pose>, Board, f64) {
    let mut x = pack(&k, poses, board);
    let n_params = x.len();
    let mut free: Vec<bool> = vec![true; n_params];
    free[1] = opts.principal_point;
    free[2] = opts.principal_point;
    free[3] = opts.distortion;
    free[4] = opts.distortion;
    for f in &mut free[5 + 6 * poses.len()..] {
        *f = opts.board_deformation;
    }
    let free_idx: Vec<usize> = (0..n_params).filter(|&i| free[i]).collect();
    let n_obs: usize = views.iter().map(|o| o.len()).sum();

    let mut r = residuals(&x, views);
    let mut cost = r.norm_squared();
    let mut lambda = 1e-3;
    for _ in 0..opts.iterations {
        let mut jac = DMatrix::<f64>::zeros(r.len(), free_idx.len());
        for (c, &pi) in free_idx.iter().enumerate() {
            let h = if pi == 0 { 1e-2 } else if pi < 5 { 1e-5 } else { 1e-6 };
            let mut xp = x.clone();
            xp[pi] += h;
            let mut xm = x.clone();
            xm[pi] -= h;
            let d = (residuals(&xp, views) - residuals(&xm, views)) / (2.0 * h);
            jac.set_column(c, &d);
        }
        let jtj = jac.transpose() * &jac;
        let jtr = jac.transpose() * &r;
        let mut improved = false;
        for _ in 0..10 {
            let mut a = jtj.clone();
            for i in 0..a.nrows() {
                a[(i, i)] += lambda * (1.0 + a[(i, i)]);
            }
            let Some(delta) = a.lu().solve(&(-&jtr)) else {
                lambda *= 10.0;
                continue;
            };
            let mut xn = x.clone();
            for (c, &pi) in free_idx.iter().enumerate() {
                xn[pi] += delta[c];
            }
            let rn = residuals(&xn, views);
            let cn = rn.norm_squared();
            if cn < cost {
                x = xn;
                r = rn;
                cost = cn;
                lambda = (lambda / 3.0).max(1e-9);
                improved = true;
                break;
            }
            lambda *= 5.0;
        }
        if !improved {
            break;
        }
    }
    let (k, poses, board) = unpack(&x, views.len());
    let rms = (r.rows(0, 2 * n_obs).norm_squared() / n_obs as f64).sqrt();
    (k, poses, board, rms)
}

/// Per-observation reprojection errors.
pub fn reprojection_errors(k: &Intrinsics, poses: &[Pose], board: &Board, views: &[Vec<Observation>]) -> Vec<Vec<f64>> {
    views
        .iter()
        .zip(poses)
        .map(|(obs, pose)| {
            obs.iter()
                .map(|o| {
                    let world = o.world + Vector3::new(0.0, 0.0, board.z_offsets[o.corner]);
                    let p = k.project(pose.to_camera(world));
                    (p[0] - o.pixel[0]).hypot(p[1] - o.pixel[1])
                })
                .collect()
        })
        .collect()
}
