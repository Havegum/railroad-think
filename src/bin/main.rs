use clap::{Args, Parser};
use game::Game;
use mcts::heuristics::Heuristics;
use mcts::MonteCarloTree;
use railroad_ink_solver::*;
use std::thread;

use rand::prelude::*;

#[derive(Parser)] // requires `derive` feature
enum Cli {
    NN(NeuralNetworkArgs),
    Play(PlayArgs),
}

#[derive(Args)]
struct NeuralNetworkArgs {
    #[arg(short, long)]
    train: bool,

    #[arg(short, long)]
    generate_training_data: bool,

    #[arg(short, long)]
    loop_training: bool,
}

#[derive(Args, Debug)]
struct PlayArgs {
    #[arg(long, default_value = "1")]
    count: u32,

    /// Seed for random number generator. If set, the games will be deterministic
    #[arg(long)]
    seed: Option<u64>,

    #[arg(short, long)]
    duration: Option<u128>,

    #[arg(short, long)]
    iterations: Option<u64>,

    #[arg(short, long)]
    loop_play: bool,
}

fn poisson(lambda: f64) -> f64 {
    let mut rng = rand::thread_rng();

    let k = poisson_lambda(lambda); // Generate Poisson-distributed integer
    let u = rng.gen::<f64>(); // Generate uniform random number between 0 and 1

    lambda + (k as f64 + u - lambda) / lambda
}

fn poisson_lambda(lambda: f64) -> u32 {
    let mut rng = rand::thread_rng();
    let l = (-lambda).exp();
    let mut p = 1.0;
    let mut k = 0;

    loop {
        k += 1;

        let u = rng.gen::<f64>(); // Generate uniform random number between 0 and 1
        p *= u;

        if p <= l {
            break;
        }
    }

    k - 1
}

fn chaos_random() -> u128 {
    let mut rng = rand::thread_rng();

    let lambda = 0.5; // Poisson parameter
    let use_poisson = rng.gen_bool(0.75);

    let sample = if use_poisson {
        poisson(lambda)
    } else {
        rng.gen_range(0.01..2.0) * rng.gen_range(0.01..2.0)
    } * 1000.;

    sample.abs() as u128
}

fn main() {
    match Cli::parse() {
        Cli::NN(args) => {
            if args.train {
                // use burn::backend::{Autodiff, Wgpu};
                // type MyBackend = Wgpu<f32, i32>;
                // type MyAutodiffBackend = Autodiff<MyBackend>;
                // let device = burn::backend::wgpu::WgpuDevice::default();

                use burn::backend::Autodiff;
                use burn_cuda::{Cuda, CudaDevice};
                type MyBackend = Cuda<f32, i32>;
                type AutodiffBackend = Autodiff<MyBackend>;
                let device = CudaDevice::default();

                // use burn::backend::Autodiff;
                // use burn::backend::NdArray;

                // type Backend = NdArray<f32>;
                // type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;
                // type AutodiffBackend = Autodiff<Backend>;

                // let device = BackendDevice::default();

                mcts::heuristics::nn::training::run::<AutodiffBackend>(device);
            } else {
                let mut initial_run = true;

                while args.loop_training || initial_run {
                    initial_run = false;

                    if args.generate_training_data {
                        let samples = 5;
                        let iterations = 200;
                        mcts::trainer::generate_training_data(samples, iterations);
                    }
                }
            }
        }
        Cli::Play(args) => {
            let mut initial_run = true;

            while args.loop_play || initial_run {
                initial_run = false;

                let play_mode = if let Some(iterations) = args.iterations {
                    PlayMode::Iterations(iterations)
                } else if let Some(duration) = args.duration {
                    PlayMode::Duration(duration)
                } else {
                    PlayMode::Duration(chaos_random())
                };

                let seed = args.seed.unwrap_or_else(|| rand::thread_rng().gen());

                // If we aren't running games in parallel, run the game on main thread,
                // for easier profiling
                if args.count == 1 {
                    let (n, score) = play(play_mode, seed.to_be_bytes());

                    match play_mode {
                        PlayMode::Iterations(_) => println!("iterations: {n}, score: {score}"),
                        PlayMode::Duration(_) => println!("{n},{score}"),
                    }
                } else {
                    let handles = run(args.count as u8, play_mode, seed);
                    for handle in handles {
                        let (n, score) = handle.join().unwrap();

                        match play_mode {
                            PlayMode::Iterations(_) => println!("iterations: {n}, score: {score}"),
                            PlayMode::Duration(_) => println!("{n},{score}"),
                        }
                    }
                }
            }
        }
    }

    // // Run the simulated annealing algorithm_
    // println!("Running simulated annealing...");
    // let max_iterations = 10_000;
    // let initial_temperature = 1.0;
    // let initial_score = 53.279;
    // let temperature_decay_rate = 0.95;
    // let variable = Some(4);
    // let heuristics = simulated_annealing(
    //     max_iterations,
    //     initial_temperature,
    //     initial_score,
    //     temperature_decay_rate,
    //     variable,
    // );
    // println!("heuristics: {heuristics:?}");

    // use mcts::heuristics::nn::edge_strategy::EdgeStrategy;
    // let mut nn = EdgeStrategy::create_model();
    // nn.train_model_path("model-16-16");

    // play(1000);
}

/// Play `n` games
/// Spawns a thread for each `n`
/// Each thread returns play mode stats and score
fn run(n: u8, play_mode: PlayMode, seed: u64) -> Vec<thread::JoinHandle<(u64, i32)>> {
    (0..n)
        .map(|i| {
            thread::spawn(move || {
                // Give each thread a unique seed, while still being determinated from the root seed
                let seed_bytes = (seed + i as u64).to_be_bytes();
                play(play_mode, seed_bytes)
            })
        })
        .collect()
}

#[derive(Copy, Clone)]
enum PlayMode {
    Iterations(u64),
    Duration(u128),
}

/// Play single game
/// Returns duration or iteration and score
fn play(play_mode: PlayMode, seed: [u8; 8]) -> (u64, i32) {
    let mut game = Game::new_from_seed(seed);

    let mut mcts = MonteCarloTree::new_from_seed(game.clone(), seed);

    // use mcts::heuristics::nn::edge_strategy::EdgeStrategy;
    // let nn = EdgeStrategy::load("model-2");

    while !game.ended {
        let mv = match play_mode {
            PlayMode::Iterations(iterations) => mcts.search_iterations(iterations).best_move(),
            PlayMode::Duration(duration) => mcts.search_duration(duration).best_move(),
        };

        // println!(
        //     "{mv}, pred: {:.1}, depth: {}",
        //     nn.predict(&game.board, &mv),
        //     mcts.calculate_depth()
        // );
        mcts = MonteCarloTree::progress(mcts, mv, &mut game);
    }

    match play_mode {
        PlayMode::Iterations(iterations) => (iterations, game.board.score()),
        PlayMode::Duration(duration) => (duration as u64, game.board.score()),
    }
}

/// Play single game
/// Returns iterations and score
pub fn play_and_dump_rave_heuristics(iterations: u64, _i: u64) -> (u64, i32) {
    let mut game = Game::new();
    game.roll();

    let heuristics = Heuristics {
        // use_rave: false,
        ..Default::default()
    };

    let mut mcts = MonteCarloTree::new_with_heuristics(game.clone(), heuristics);

    while !game.ended {
        let mv = mcts.search_iterations(iterations).best_move();
        println!("best move: {mv:?}");
        mcts = MonteCarloTree::progress(mcts, mv, &mut game);
    }

    (iterations, game.board.score())
}
