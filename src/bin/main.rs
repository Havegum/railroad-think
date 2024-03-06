#![feature(stdsimd)]

use clap::{Args, Parser};
use game::Game;
use mcts::heuristics::Heuristics;
use mcts::MonteCarloTree;
// use railroad_ink_solver::mcts::heuristics;
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
    #[arg(short, long, default_value_t = true)]
    train: bool,

    #[arg(short, long)]
    generate_training_data: bool,

    #[arg(short, long)]
    loop_training: bool,

    #[arg(long)]
    create_model: bool,
}

#[derive(Args, Debug)]
struct PlayArgs {
    #[arg(long, default_value = "1")]
    count: u32,

    #[arg(short, long, default_value = "1000")]
    duration: Option<u128>,

    #[arg(short, long)]
    iterations: Option<u64>,

    #[arg(short, long)]
    loop_play: bool,

    #[arg(short, long)]
    sample_duration: bool,
}

fn chaos_random() -> u128 {
    let mut rng = rand::thread_rng();

    // let lambda = 0.1; // Poisson parameter
    // let use_poisson = rng.gen_bool(0.75);

    // let sample = if use_poisson {
    // let sample = {
    // poisson(lambda)
    // } else {
    // rng.gen_range(0.01..1.8) * rng.gen_range(0.01..1.8)

    // }
    let sample: f64 = rng.gen_range(0.001..1.0) * rng.gen_range(0.001..1.0) * 2000.0;
    sample.abs() as u128
}

fn main() {
    match Cli::parse() {
        Cli::NN(args) => {
            let mut initial_run = true;

            let mut model = if args.create_model {
                mcts::heuristics::nn::edge_strategy::EdgeStrategy::create_model()
            } else {
                mcts::heuristics::nn::edge_strategy::EdgeStrategy::load("model-conv")
            };

            while args.loop_training || initial_run {
                initial_run = false;

                if args.generate_training_data {
                    let samples = 5;
                    let iterations = 500;
                    mcts::trainer::generate_training_data(samples, iterations);
                }

                if args.train {
                    model.train_model_path("model-conv", 100);
                }
            }
        }
        Cli::Play(args) => {
            let mut initial_run = true;

            while args.loop_play || initial_run {
                initial_run = false;

                let play_mode = if args.sample_duration {
                    PlayMode::Duration(chaos_random())
                } else {
                    if let Some(iterations) = args.iterations {
                        PlayMode::Iterations(iterations)
                    } else {
                        PlayMode::Duration(args.duration.unwrap())
                    }
                };

                let all_heuristics = vec![
                    String::from("./config/heuristics.json"),
                    String::from("./config/heuristics-0.json"),
                    String::from("./config/heuristics-4.json"),
                    String::from("./config/heuristics-8.json"),
                    String::from("./config/heuristics-12.json"),
                    String::from("./config/heuristics-16.json"),
                    String::from("./config/heuristics-20.json"),
                ];

                for heuristics in all_heuristics {
                    let handles = run(args.count as u8, play_mode, heuristics.clone());
                    for handle in handles {
                        let (n, score) = handle.join().unwrap();

                        match play_mode {
                            PlayMode::Iterations(_) => println!("iterations: {n}, score: {score}"),
                            PlayMode::Duration(_) => println!("{heuristics},{n},{score}"),
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
fn run(n: u8, play_mode: PlayMode, heuristics: String) -> Vec<thread::JoinHandle<(u64, i32)>> {
    (0..n)
        .map(|_| heuristics.clone())
        .map(|heuristics| thread::spawn(move || play(play_mode, heuristics)))
        .collect()
}

#[derive(Copy, Clone)]
enum PlayMode {
    Iterations(u64),
    Duration(u128),
}

/// Play single game
/// Returns duration or iteration and score
fn play(play_mode: PlayMode, heuristics: String) -> (u64, i32) {
    let mut game = Game::new();

    let heuristics = Heuristics::from_json(heuristics.as_str()).expect("Could not parse json");
    let mut mcts = MonteCarloTree::new_with_heuristics(game.clone(), heuristics);

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
