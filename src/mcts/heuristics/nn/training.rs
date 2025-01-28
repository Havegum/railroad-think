use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use std::fs;

use super::{
    data::{DataBatcher, GameDataset},
    Model, ModelConfig,
};

static ARTIFACT_DIR: &str = "./tmp/nn-data";
static MODEL_OUTPUT_DIR: &str = "./src/mcts/heuristics/nn";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,

    #[config(default = 50)]
    pub num_epochs: usize,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// # Panics
///
/// This function panics if the model cannot be saved
pub fn run<B: AutodiffBackend>(device: &B::Device) {
    let mut checkpoint = if std::path::Path::new(&format!("{ARTIFACT_DIR}/checkpoint")).exists() {
        fs::read_dir(format!("{ARTIFACT_DIR}/checkpoint"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter_map(|dir_entry| dir_entry.file_name().into_string().ok())
            .filter(|name| name.starts_with("model-") && name.contains('.'))
            .map(|string| string.as_str()[6..].to_string())
            .map(|string| string.split('.').next().unwrap().to_string())
            .filter_map(|string| string.parse::<usize>().ok())
            .fold(0, std::cmp::Ord::max)
    } else {
        0
    };

    let use_checkpoint = checkpoint > 0;

    if use_checkpoint {
        for file in ["model", "optim", "scheduler"] {
            fs::rename(
                format!("{ARTIFACT_DIR}/checkpoint/{file}-{checkpoint}.mpk"),
                format!("./tmp/{file}.mpk"),
            )
            .expect("Could not move checkpoint to temporary dir");
        }
    }

    create_artifact_dir(&format!("{ARTIFACT_DIR}/checkpoint"));

    if use_checkpoint {
        for file in ["model", "optim", "scheduler"] {
            fs::rename(
                format!("./tmp/{file}.mpk"),
                format!("{ARTIFACT_DIR}/checkpoint/{file}-1.mpk"),
            )
            .unwrap_or_else(|_| {
                panic!(
                    "Could not move checkpoint (./tmp/{}-{}.mpk) from temporary dir",
                    file, checkpoint
                )
            });
        }
        checkpoint = 1;
    }

    // Config
    let config_optimizer = AdamConfig::new();
    let config = TrainingConfig::new(ModelConfig::new(), config_optimizer).with_num_epochs(1);
    B::seed(config.seed);

    // Data
    let batcher_train = DataBatcher::<B>::new(device.clone());
    let batcher_valid = DataBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(GameDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(GameDataset::test());

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 2 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary();

    let learner = if use_checkpoint {
        learner.checkpoint(checkpoint)
    } else {
        learner
    };

    let learner = learner.build(Model::init(device), config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{MODEL_OUTPUT_DIR}/model.config.json").as_str())
        .unwrap();

    model_trained
        .save_file(format!("{MODEL_OUTPUT_DIR}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
