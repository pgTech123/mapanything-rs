//! Encoder of a global (per-view) quantity such as a camera rotation, translation or scale
//! (UniCeption's GlobalRepresentationEncoder): an MLP to the token width, then a LayerNorm.
//! MapAnything adds its output to every patch token of the view.

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::dinov2::DIM;
use crate::vit_block::layer_norm;

const HIDDEN: [usize; 3] = [128, 256, 512];

pub struct GlobalEncoder {
    layers: [Linear; 4],
    norm: LayerNorm,
}

impl GlobalEncoder {
    /// `vb` at the encoder (e.g. `cam_rot_encoder`); `in_dim` is 4 for quaternions, 3 for
    /// translations, 1 for a log scale.
    pub fn new(in_dim: usize, vb: VarBuilder) -> Result<Self> {
        // The checkpoint keeps the nested nn.Sequential numbering.
        Ok(Self {
            layers: [
                candle_nn::linear(in_dim, HIDDEN[0], vb.pp("encoder.0.0.0.0"))?,
                candle_nn::linear(HIDDEN[0], HIDDEN[1], vb.pp("encoder.0.0.1"))?,
                candle_nn::linear(HIDDEN[1], HIDDEN[2], vb.pp("encoder.0.1"))?,
                candle_nn::linear(HIDDEN[2], DIM, vb.pp("encoder.1"))?,
            ],
            norm: layer_norm(DIM, vb.pp("norm_layer"))?,
        })
    }

    /// `input`: [1, in_dim]. Returns [1, DIM].
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut xs = input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if index + 1 < self.layers.len() {
                xs = xs.gelu_erf()?;
            }
        }
        self.norm.forward(&xs)
    }
}
