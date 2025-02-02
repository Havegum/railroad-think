use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig};
use burn::nn::Gelu;
use burn::prelude::{Backend, Module, Tensor};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    pool: AvgPool2d,
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
        let pool = AvgPool2dConfig::new(kernel_size).init();
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
