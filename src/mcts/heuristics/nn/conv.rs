use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AvgPool2d, MaxPool2d, MaxPool2dConfig};
// use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Gelu};
use burn::prelude::{Backend, Module, Tensor};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    pool: MaxPool2d,
    // norm: BatchNorm<B, 2>,
    activation: Gelu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], kernel_size).init(device);
        let pool = MaxPool2dConfig::new(kernel_size).init();
        // let norm = BatchNormConfig::new(out_channels).init(device);
        Self {
            conv,
            pool,
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.pool.forward(x);
        self.activation.forward(x)
    }
}
