use burn::nn::loss::{MseLoss, Reduction};
use burn::prelude::{Backend, Config, Module, Tensor};

mod conv;
pub mod data;
mod linear;
pub mod training;

use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use conv::ConvBlock;
use data::DataBatch;
use linear::LinearBlock;

#[derive(Config, Debug)]
pub struct ModelConfig {}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_b_size = 11;

        let conv_block1 = ConvBlock::init(12, 7, [3, 3], device);
        let conv_block2 = ConvBlock::init(7, 7, [3, 3], device);
        let linear_block1 = LinearBlock::init(7 + input_b_size, 64, device);
        let linear_block2 = LinearBlock::init(64, 32, device);
        let output_block = LinearBlock::init(32, 1, device);

        Model {
            conv_block1,
            conv_block2,
            linear_block1,
            linear_block2,
            output_block,
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv_block1: ConvBlock<B>,
    conv_block2: ConvBlock<B>,
    linear_block1: LinearBlock<B>,
    linear_block2: LinearBlock<B>,
    output_block: LinearBlock<B>,
}

impl<B: Backend> Model<B> {
    pub fn init(device: &B::Device) -> Self {
        let config = ModelConfig {};
        config.init(device)
    }

    pub fn forward(&self, input_a: Tensor<B, 4>, input_b: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, _] = input_b.dims();
        let x = input_a.swap_dims(1, 3);

        let x = self.conv_block1.forward(x);
        let x = self.conv_block2.forward(x);
        let [_, dim_x, dim_y, dim_z] = x.dims();
        let x = x.reshape([batch_size, dim_x * dim_y * dim_z]); // Flatten the tensor
        let x = Tensor::cat(vec![x, input_b], 1); // Concatenate along the feature dimension
        let x = self.linear_block1.forward(x);
        let x = self.linear_block2.forward(x);
        self.output_block.forward(x)
    }

    pub fn forward_step(&self, item: DataBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze_dim(1);
        let output = self.forward(item.boards, item.heuristics);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<DataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: DataBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: DataBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
