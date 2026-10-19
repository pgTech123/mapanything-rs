//! Photo loading: EXIF orientation and focal length, downscaling to the working resolution,
//! grayscale conversion.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

/// Diagonal of a 36 x 24 mm frame, the reference of 35 mm-equivalent focal lengths.
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615;

/// A grayscale image with f32 pixels in [0, 1], pixel centers at integer coordinates.
#[derive(Clone)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Bilinear sample; None outside the image.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0 && x <= (self.width - 1) as f64 && y <= (self.height - 1) as f64) {
            return None;
        }
        let x0 = (x.floor() as usize).min(self.width - 2);
        let y0 = (y.floor() as usize).min(self.height - 2);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;
        let i = y0 * self.width + x0;
        let top = self.data[i] * (1.0 - fx) + self.data[i + 1] * fx;
        let bottom = self.data[i + self.width] * (1.0 - fx) + self.data[i + self.width + 1] * fx;
        Some(top * (1.0 - fy) + bottom * fy)
    }
}

pub struct Photo {
    pub path: PathBuf,
    /// Downscaled to the working resolution.
    pub rgb: RgbImage,
    pub gray: GrayImage,
    /// Pinhole focal length in pixels at the working resolution, from EXIF.
    pub focal_px: Option<f64>,
    /// Size of the upright photo before downscaling.
    pub full_size: (u32, u32),
}

fn read_upright(path: &Path) -> Result<DynamicImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("decoding {}", path.display()))?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

fn exif_focal_35mm(path: &Path) -> Option<f64> {
    let file = std::fs::File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()?;
    let field = exif.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)?;
    field.value.get_uint(0).map(|mm| mm as f64)
}

pub fn to_gray(rgb: &RgbImage) -> GrayImage {
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let data = rgb
        .pixels()
        .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
        .collect();
    GrayImage { width, height, data }
}

/// Loads a photo upright and shrinks it by `downscale`.
pub fn load_photo(path: &Path, downscale: u32, focal_override: Option<f64>) -> Result<Photo> {
    let full = read_upright(path)?;
    let (full_w, full_h) = (full.width(), full.height());
    let (w, h) = ((full_w / downscale).max(1), (full_h / downscale).max(1));
    let rgb = full.resize_exact(w, h, FilterType::Triangle).to_rgb8();
    let focal_35mm = focal_override.or_else(|| exif_focal_35mm(path));
    let focal_px = focal_35mm.map(|f35| {
        f35 * (full_w as f64).hypot(full_h as f64) / FULL_FRAME_DIAGONAL_MM * (w as f64 / full_w as f64)
    });
    let gray = to_gray(&rgb);
    Ok(Photo { path: path.to_path_buf(), rgb, gray, focal_px, full_size: (full_w, full_h) })
}
