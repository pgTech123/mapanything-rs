//! Grouping detected corners into the checkerboard's lattice and placing that lattice on
//! the physical board, even when only part of the board is visible.
//!
//! 1. A grid of integer (i, j) cells is grown from a seed corner by extrapolating steps to
//!    neighbors.
//! 2. A homography fitted to the grid predicts every other lattice position and pulls in
//!    the corners found there, which bridges occluded stretches. Only corners sitting
//!    between black and white squares in the expected pattern are kept.
//! 3. The lattice is placed on the board: the four orientations compatible with a camera in
//!    front of the board are tried, the board's edges are looked for by probing where the
//!    squares give way to paper, and the checker colors fix the remaining parity. A unique
//!    placement is required.

use std::collections::HashMap;

use nalgebra::Matrix3;

use crate::board::corners::Corner;
use crate::board::geometry::{apply_homography, homography_dlt};
use crate::board::photo::GrayImage;

/// Corners placed on the board: inner corner (i, j) (i along the long axis) -> pixel.
pub type Lattice = HashMap<(i32, i32), [f64; 2]>;

/// Inner corner counts of the board: (long axis, short axis).
pub type BoardSize = (i32, i32);

const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn snap(corners: &[Corner], used: &[bool], pred: [f64; 2], tol: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (k, c) in corners.iter().enumerate() {
        if used[k] {
            continue;
        }
        let d = dist(c.pos, pred);
        if d < tol && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((k, d));
        }
    }
    best.map(|(k, _)| k)
}

fn grow_from_seed(corners: &[Corner], seed: usize) -> Option<HashMap<(i32, i32), usize>> {
    let p0 = corners[seed].pos;
    let mut others: Vec<(usize, f64)> = corners
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != seed)
        .map(|(k, c)| (k, dist(c.pos, p0)))
        .collect();
    others.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (n1, d1) = *others.first()?;
    let e1 = [corners[n1].pos[0] - p0[0], corners[n1].pos[1] - p0[1]];
    let n2 = others.iter().find(|(k, d)| {
        let e2 = [corners[*k].pos[0] - p0[0], corners[*k].pos[1] - p0[1]];
        let cos = (e1[0] * e2[0] + e1[1] * e2[1]) / (d1 * d);
        *d > 0.4 * d1 && *d < 2.5 * d1 && cos.abs() < 0.8
    })?;
    let n2 = n2.0;

    let mut cells: HashMap<(i32, i32), usize> = HashMap::new();
    let mut used = vec![false; corners.len()];
    for (cell, k) in [((0, 0), seed), ((1, 0), n1), ((0, 1), n2)] {
        cells.insert(cell, k);
        used[k] = true;
    }
    let mut queue = vec![(0, 0), (1, 0), (0, 1)];
    while let Some((i, j)) = queue.pop() {
        let p = corners[cells[&(i, j)]].pos;
        for (di, dj) in DIRS {
            let target = (i + di, j + dj);
            if cells.contains_key(&target) {
                continue;
            }
            let pred = if let Some(&prev) = cells.get(&(i - di, j - dj)) {
                let pp = corners[prev].pos;
                Some([2.0 * p[0] - pp[0], 2.0 * p[1] - pp[1]])
            } else {
                let (oi, oj) = (dj, di);
                [(oi, oj), (-oi, -oj)].into_iter().find_map(|(oi, oj)| {
                    let a = cells.get(&(i + oi, j + oj))?;
                    let b = cells.get(&(i + oi + di, j + oj + dj))?;
                    let (pa, pb) = (corners[*a].pos, corners[*b].pos);
                    Some([p[0] + pb[0] - pa[0], p[1] + pb[1] - pa[1]])
                })
            };
            let Some(pred) = pred else { continue };
            let step = dist(pred, p);
            if let Some(k) = snap(corners, &used, pred, 0.3 * step) {
                cells.insert(target, k);
                used[k] = true;
                queue.push(target);
            }
        }
    }
    Some(cells)
}

fn fit_cells(corners: &[Corner], cells: &HashMap<(i32, i32), usize>) -> Option<Matrix3<f64>> {
    let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) = cells
        .iter()
        .map(|(&(i, j), &k)| ([i as f64, j as f64], corners[k].pos))
        .unzip();
    homography_dlt(&src, &dst)
}

fn fit_lattice(lattice: &Lattice) -> Option<Matrix3<f64>> {
    let (src, dst): (Vec<[f64; 2]>, Vec<[f64; 2]>) =
        lattice.iter().map(|(&(i, j), &p)| ([i as f64, j as f64], p)).unzip();
    homography_dlt(&src, &dst)
}

fn local_step(h: &Matrix3<f64>, i: i32, j: i32) -> f64 {
    let p = apply_homography(h, [i as f64, j as f64]);
    let a = apply_homography(h, [i as f64 + 1.0, j as f64]);
    let b = apply_homography(h, [i as f64, j as f64 + 1.0]);
    dist(p, a).min(dist(p, b))
}

/// Mean of the two diagonal cells minus the mean of the other two around lattice corner
/// (i, j); None when a cell is off the image or the cells of a diagonal disagree.
fn checker_contrast(h: &Matrix3<f64>, gray: &GrayImage, i: i32, j: i32) -> Option<f64> {
    let cell = |di: f64, dj: f64| {
        let p = apply_homography(h, [i as f64 + di, j as f64 + dj]);
        gray.sample(p[0], p[1]).map(|g| g as f64)
    };
    let (c00, c11, c10, c01) = (cell(-0.5, -0.5)?, cell(0.5, 0.5)?, cell(0.5, -0.5)?, cell(-0.5, 0.5)?);
    if (c00 - c11).abs() > 0.25 || (c10 - c01).abs() > 0.25 {
        return None;
    }
    Some((c00 + c11 - c10 - c01) / 2.0)
}

/// +1 when cells with even i+j are white (the contrast at corners with even i+j is
/// positive), -1 when they are black, by vote. Cell (i, j) lies between corners i..i+1 and
/// j..j+1; the two cells on the diagonal of corner (i, j) are (i-1, j-1) and (i, j), both
/// of parity i+j.
fn checker_parity(h: &Matrix3<f64>, gray: &GrayImage, cells: impl Iterator<Item = (i32, i32)>) -> f64 {
    let vote: f64 = cells
        .filter_map(|(i, j)| checker_contrast(h, gray, i, j).map(|c| c * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }))
        .sum();
    if vote >= 0.0 { 1.0 } else { -1.0 }
}

const MIN_CHECKER_CONTRAST: f64 = 0.12;

fn is_checker_corner(h: &Matrix3<f64>, gray: &GrayImage, parity: f64, i: i32, j: i32) -> bool {
    let Some(c) = checker_contrast(h, gray, i, j) else { return false };
    let expected = parity * if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
    c * expected > MIN_CHECKER_CONTRAST
}

fn complete_with_homography(corners: &[Corner], gray: &GrayImage, cells: &mut HashMap<(i32, i32), usize>) {
    for _ in 0..20 {
        if cells.len() < 6 {
            return;
        }
        let Some(h) = fit_cells(corners, cells) else { return };
        let parity = checker_parity(&h, gray, cells.keys().copied());
        cells.retain(|&(i, j), &mut k| {
            let pred = apply_homography(&h, [i as f64, j as f64]);
            dist(pred, corners[k].pos) < 0.15 * local_step(&h, i, j) && is_checker_corner(&h, gray, parity, i, j)
        });
        if cells.len() < 6 {
            return;
        }
        let mut used = vec![false; corners.len()];
        cells.values().for_each(|&k| used[k] = true);
        let (imin, imax) = cells.keys().fold((i32::MAX, i32::MIN), |(a, b), &(i, _)| (a.min(i), b.max(i)));
        let (jmin, jmax) = cells.keys().fold((i32::MAX, i32::MIN), |(a, b), &(_, j)| (a.min(j), b.max(j)));
        let mut added = 0;
        for i in imin - 2..=imax + 2 {
            for j in jmin - 2..=jmax + 2 {
                if cells.contains_key(&(i, j)) || !is_checker_corner(&h, gray, parity, i, j) {
                    continue;
                }
                let pred = apply_homography(&h, [i as f64, j as f64]);
                if let Some(k) = snap(corners, &used, pred, 0.2 * local_step(&h, i, j)) {
                    cells.insert((i, j), k);
                    used[k] = true;
                    added += 1;
                }
            }
        }
        if added == 0 {
            return;
        }
    }
}

/// The largest lattice among `corners`, with integer coordinates relative to an arbitrary
/// origin and orientation, as pixel positions.
pub fn find_lattice(corners: &[Corner], gray: &GrayImage) -> Option<Lattice> {
    let mut best: Option<HashMap<(i32, i32), usize>> = None;
    for seed in 0..corners.len().min(80) {
        let Some(mut cells) = grow_from_seed(corners, seed) else { continue };
        if cells.len() < 6 {
            continue;
        }
        complete_with_homography(corners, gray, &mut cells);
        if best.as_ref().is_none_or(|b| cells.len() > b.len()) {
            best = Some(cells);
        }
    }
    best.map(|cells| cells.into_iter().map(|(ij, k)| (ij, corners[k].pos)).collect())
}

pub fn extents(lattice: &Lattice) -> (i32, i32) {
    let (imin, imax) = lattice.keys().fold((i32::MAX, i32::MIN), |(a, b), &(i, _)| (a.min(i), b.max(i)));
    let (jmin, jmax) = lattice.keys().fold((i32::MAX, i32::MIN), |(a, b), &(_, j)| (a.min(j), b.max(j)));
    (imax - imin + 1, jmax - jmin + 1)
}

/// What a row of cells beyond the lattice looks like.
#[derive(PartialEq, Debug)]
enum RowKind {
    /// Alternating black and white squares in the expected pattern: the board continues.
    Squares,
    /// Uniformly bright: the paper margin, the board has ended.
    Paper,
    /// Off the image, covered, or something else.
    Unknown,
}

/// Gray levels of the board's black and white squares, as medians over the lattice's cells.
fn board_levels(h: &Matrix3<f64>, gray: &GrayImage, parity: f64, lattice: &Lattice) -> Option<(f64, f64)> {
    let (mut blacks, mut whites) = (Vec::new(), Vec::new());
    for &(i, j) in lattice.keys() {
        // The cell with corner (i, j) as its min corner.
        let p = apply_homography(h, [i as f64 + 0.5, j as f64 + 0.5]);
        let Some(g) = gray.sample(p[0], p[1]) else { continue };
        let even = (i + j).rem_euclid(2) == 0;
        if (parity > 0.0) == even { whites.push(g as f64) } else { blacks.push(g as f64) }
    }
    if blacks.len() < 3 || whites.len() < 3 {
        return None;
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let (b, w) = (median(&mut blacks), median(&mut whites));
    (w - b > 0.15).then_some((b, w))
}

/// Mean gray of a small cross of samples around lattice point (ci, cj); None if any sample
/// is off the image.
fn sample_cross(h: &Matrix3<f64>, gray: &GrayImage, ci: f64, cj: f64) -> Option<f64> {
    let step = 0.05;
    let mut sum = 0.0;
    for (di, dj) in [(0.0, 0.0), (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
        let q = apply_homography(h, [ci + di, cj + dj]);
        sum += gray.sample(q[0], q[1])? as f64;
    }
    Some(sum / 5.0)
}

/// Classifies cell row `k` along `axis` (0 = i, 1 = j) over the lattice's span on the
/// other axis. `facing` is the row's edge facing the lattice (0 or 1): paper is looked for
/// just beyond that edge, so that a narrow paper margin is still hit; squares are looked
/// for at the cells' centers.
#[allow(clippy::too_many_arguments)]
fn classify_row(h: &Matrix3<f64>, gray: &GrayImage, parity: f64, levels: (f64, f64), axis: usize, k: i32, span: (i32, i32), facing: f64) -> RowKind {
    let (black, white) = levels;
    let is_bright = |g: f64| g > black + 0.65 * (white - black);
    let is_dark = |g: f64| g < black + 0.35 * (white - black);
    let edge_inset = if facing == 0.0 { PROBE_INSET } else { 1.0 - PROBE_INSET };
    let (mut edge_bright, mut edge_dark, mut edge_visible) = (0, 0, 0);
    let (mut matched, mut dark, mut center_visible, mut total) = (0, 0, 0, 0);
    for m in span.0..span.1 {
        total += 1;
        let at = |inset: f64| if axis == 0 { (k as f64 + inset, m as f64 + 0.5) } else { (m as f64 + 0.5, k as f64 + inset) };
        let (ei, ej) = at(edge_inset);
        if let Some(g) = sample_cross(h, gray, ei, ej) {
            edge_visible += 1;
            if is_bright(g) {
                edge_bright += 1;
            } else if is_dark(g) {
                edge_dark += 1;
            }
        }
        let (ci, cj) = at(0.5);
        if let Some(g) = sample_cross(h, gray, ci, cj) {
            center_visible += 1;
            let (cell_i, cell_j) = if axis == 0 { (k, m) } else { (m, k) };
            let expect_black = parity * if (cell_i + cell_j).rem_euclid(2) == 0 { 1.0 } else { -1.0 } < 0.0;
            if is_dark(g) {
                dark += 1;
                if expect_black {
                    matched += 1;
                }
            } else if is_bright(g) && !expect_black {
                matched += 1;
            }
        }
    }
    let enough = |visible: usize| visible * 10 >= total * 7 && visible >= 2;
    let kind = if enough(center_visible) && dark >= 1 && matched * 10 >= center_visible * 8 {
        RowKind::Squares
    } else if enough(edge_visible) && edge_dark == 0 && edge_bright * 10 >= edge_visible * 8 {
        RowKind::Paper
    } else {
        RowKind::Unknown
    };
    if std::env::var_os("BOARD_POSE_DEBUG").is_some() {
        eprintln!(
            "      axis {axis} cell row {k}: {kind:?} (edge: {edge_bright} bright, {edge_dark} dark of {edge_visible}/{total}; center: {matched} matched, {dark} dark of {center_visible})"
        );
    }
    kind
}

/// How far into a cell row, from the side facing the lattice, rows beyond the lattice are
/// sampled: small enough to land in a narrow paper margin.
const PROBE_INSET: f64 = 0.12;

/// Index of the first and last inner corner along `axis`, when the board's edge is visible
/// on that side: squares are followed by a row of plain paper.
fn probe_ends(h: &Matrix3<f64>, gray: &GrayImage, parity: f64, lattice: &Lattice, axis: usize) -> (Option<i32>, Option<i32>) {
    let Some(levels) = board_levels(h, gray, parity, lattice) else { return (None, None) };
    let (lo, hi) = lattice.keys().fold((i32::MAX, i32::MIN), |(a, b), &(i, j)| {
        let v = if axis == 0 { i } else { j };
        (a.min(v), b.max(v))
    });
    let (slo, shi) = lattice.keys().fold((i32::MAX, i32::MIN), |(a, b), &(i, j)| {
        let v = if axis == 0 { j } else { i };
        (a.min(v), b.max(v))
    });
    let span = (slo, shi); // cells between corners slo..shi
    let mut first = None;
    let mut last = None;
    // Max side: cell row k lies between corners k and k+1. Paper at k means corner row k
    // is the outer edge, so the last inner corner is k-1.
    for k in hi + 1..hi + 40 {
        match classify_row(h, gray, parity, levels, axis, k, span, 0.0) {
            RowKind::Squares => continue,
            RowKind::Paper => {
                last = Some(k - 1);
                break;
            }
            RowKind::Unknown => break,
        }
    }
    // Min side: paper at cell row k means corner row k+1 is the edge; first inner is k+2.
    for k in (lo - 40..=lo - 2).rev() {
        match classify_row(h, gray, parity, levels, axis, k, span, 1.0) {
            RowKind::Squares => continue,
            RowKind::Paper => {
                first = Some(k + 2);
                break;
            }
            RowKind::Unknown => break,
        }
    }
    (first, last)
}

/// Lattice transformed by one of the eight orientations of the square lattice: a quarter
/// turn `rotate` times, after optionally transposing the axes.
fn transform(lattice: &Lattice, transpose: bool, rotate: u8) -> Lattice {
    lattice
        .iter()
        .map(|(&(i, j), &p)| {
            let (mut a, mut b) = if transpose { (j, i) } else { (i, j) };
            for _ in 0..rotate {
                (a, b) = (-b, a);
            }
            ((a, b), p)
        })
        .collect()
}

/// Possible offsets along one axis given the lattice's range, the board's corner count
/// and the edges found.
fn axis_offsets(range: (i32, i32), n: i32, ends: (Option<i32>, Option<i32>)) -> Vec<i32> {
    let (lo, hi) = range;
    if hi - lo + 1 > n {
        return Vec::new();
    }
    // lo + o >= 0 and hi + o <= n - 1
    (-lo..=n - 1 - hi)
        .filter(|&o| ends.0.is_none_or(|f| f + o == 0) && ends.1.is_none_or(|l| l + o == n - 1))
        .collect()
}

pub struct Placement {
    pub lattice: Lattice,
    /// How the placement was decided, for the debug output.
    pub note: String,
}

/// Places a lattice on the board. Err describes why it could not be done uniquely.
pub fn place_on_board(lattice: &Lattice, board: BoardSize, gray: &GrayImage) -> Result<Placement, String> {
    let (n_long, n_short) = board;
    let debug = std::env::var_os("BOARD_POSE_DEBUG").is_some();
    let mut solutions: Vec<(Lattice, String)> = Vec::new();
    for transpose in [false, true] {
        for rotate in 0..4u8 {
            let lat = transform(lattice, transpose, rotate);
            let Some(h) = fit_lattice(&lat) else { continue };
            // Handedness: with image y down and the camera in front of the board, the
            // cross product of the i and j steps is negative.
            let center = {
                let (ni, nj) = extents(&lat);
                let imin = lat.keys().map(|k| k.0).min().unwrap();
                let jmin = lat.keys().map(|k| k.1).min().unwrap();
                [imin as f64 + (ni - 1) as f64 / 2.0, jmin as f64 + (nj - 1) as f64 / 2.0]
            };
            let p = apply_homography(&h, center);
            let a = apply_homography(&h, [center[0] + 1.0, center[1]]);
            let b = apply_homography(&h, [center[0], center[1] + 1.0]);
            if (a[0] - p[0]) * (b[1] - p[1]) - (a[1] - p[1]) * (b[0] - p[0]) > 0.0 {
                continue;
            }
            let parity = checker_parity(&h, gray, lat.keys().copied());
            let ends_i = probe_ends(&h, gray, parity, &lat, 0);
            let ends_j = probe_ends(&h, gray, parity, &lat, 1);
            let range = |axis: usize| {
                lat.keys().fold((i32::MAX, i32::MIN), |(a, b), &(i, j)| {
                    let v = if axis == 0 { i } else { j };
                    (a.min(v), b.max(v))
                })
            };
            if debug {
                eprintln!(
                    "    transpose {transpose} rotate {rotate}: i range {:?} ends {:?}, j range {:?} ends {:?}, parity {parity}",
                    range(0),
                    ends_i,
                    range(1),
                    ends_j
                );
            }
            for oi in axis_offsets(range(0), n_long, ends_i) {
                for oj in axis_offsets(range(1), n_short, ends_j) {
                    // Cell (0,0) of the board is black, so in board coordinates even cells
                    // are black: lattice parity flips when the offset sum is odd.
                    let board_parity = parity * if (oi + oj).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                    if board_parity > 0.0 {
                        if debug {
                            eprintln!("    transpose {transpose} rotate {rotate}: offsets ({oi}, {oj}) rejected by checker colors");
                        }
                        continue;
                    }
                    let placed: Lattice = lat.iter().map(|(&(i, j), &p)| ((i + oi, j + oj), p)).collect();
                    let edges = [ends_i.0.is_some(), ends_i.1.is_some(), ends_j.0.is_some(), ends_j.1.is_some()]
                        .iter()
                        .filter(|&&e| e)
                        .count();
                    solutions.push((placed, format!("{edges} board edges seen")));
                }
            }
        }
    }
    match solutions.len() {
        1 => {
            let (lattice, note) = solutions.pop().unwrap();
            Ok(Placement { lattice, note })
        }
        0 => Err(format!(
            "no placement fits a {n_long}x{n_short} board (lattice {}x{})",
            extents(lattice).0,
            extents(lattice).1
        )),
        n => Err(format!(
            "{n} possible placements (lattice {}x{}, too few board edges visible)",
            extents(lattice).0,
            extents(lattice).1
        )),
    }
}
