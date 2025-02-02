use crate::board::Board;
use crate::game::mv::Move;
use crate::game::Game;
use crate::mcts::Score;
use crate::pieces::Piece;
use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    config::Config,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use nn::{data::DataItem, training::TrainingConfig, Model};
use ord_subset::OrdSubsetIterExt;
use std::fs::File;
use std::io::prelude::*;
pub mod nn;

pub type HeuristicOptions = [[f64; 7]; 8];
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Parameters {
    pub unexplored_value: [f64; 7],
    pub exploration_variables: [f64; 7],
    pub special_cost: [f64; 7],
    pub piece_connects_to_exit: [f64; 7],
    pub piece_connects_to_other_piece: [f64; 7],
    pub piece_locks_out_other_piece: [f64; 7],
    pub piece_is_2nd_order_neighbor: [f64; 7],
    pub piece_is_3rd_order_neighbor: [f64; 7],
    pub prune_minimum_node_count: u16,
    pub prune_alpha: f64,
}

/// TODO: make `from_json` and `to_json`, and make the appropriate json
// https://docs.rs/serde_json/latest/serde_json/
impl Parameters {
    /// Save the heuristic parameters to a json file.
    /// # Errors
    /// Returns an error if the file cannot be opened or written to.
    pub fn to_json(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        let contents = serde_json::to_string(self)?;
        write!(file, "{contents}")?;
        Ok(())
    }

    /// Load the heuristic parameters from a json file.
    ///
    /// # Panics
    /// Panics if the file cannot be read.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened.
    pub fn from_json(path: &str) -> Result<Self, String> {
        File::open(path).map_or_else(
            |_| Err("Error loading Heuristics: Could not find path".to_string()),
            |mut file| {
                let mut contents = String::new();

                file.read_to_string(&mut contents)
                    .expect("Error loading Heuristics: Could not read file to string");

                match serde_json::from_str(&contents) {
                    Ok(parameters) => Ok(parameters),
                    Err(e) => Err(format!("Error loading Heuristics: {e}")),
                }
            },
        )
    }

    #[must_use]
    pub const fn as_array(&self) -> [[f64; 7]; 8] {
        [
            self.unexplored_value,
            self.exploration_variables,
            self.special_cost,
            self.piece_connects_to_exit,
            self.piece_connects_to_other_piece,
            self.piece_locks_out_other_piece,
            self.piece_is_2nd_order_neighbor,
            self.piece_is_3rd_order_neighbor,
        ]
    }
}

impl From<[[f64; 7]; 8]> for Parameters {
    fn from(array: [[f64; 7]; 8]) -> Self {
        Self {
            unexplored_value: array[0],
            exploration_variables: array[1],
            special_cost: array[2],
            piece_connects_to_exit: array[3],
            piece_connects_to_other_piece: array[4],
            piece_locks_out_other_piece: array[5],
            piece_is_2nd_order_neighbor: array[6],
            piece_is_3rd_order_neighbor: array[7],
            prune_minimum_node_count: 60,
            prune_alpha: 4.0,
        }
    }
}

#[derive(Clone)]
pub struct Heuristics {
    pub parameters: Parameters,
    pub tree_reuse: bool,
    pub move_nn: Option<Model<Wgpu>>,
}

impl Heuristics {
    #[must_use]
    pub const fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            move_nn: None,
            tree_reuse: true,
        }
    }

    #[must_use]
    /// # Panics
    /// Panics if the model files can't be found, or if the trained model does not match the config.
    pub fn get_default_model(device: &WgpuDevice) -> Model<Wgpu> {
        const MODEL_DIR: &str = "./src/mcts/heuristics/nn";
        let config = TrainingConfig::load(format!("{MODEL_DIR}/model.config.json"))
            .expect("Config should exist for the model");
        let record = CompactRecorder::new()
            .load(format!("{MODEL_DIR}/model").into(), device)
            .expect("Trained model should exist");

        config.model.init(device).load_record(record)
    }

    #[must_use]
    pub fn with_model(mut self, device: &WgpuDevice) -> Self {
        self.move_nn = Some(Self::get_default_model(device));
        self
    }

    /// Export this instance of Heuristics to a `.json`-file at the given `path`
    /// # Errors
    /// Erros if there was an error writing to the file
    /// # Panics
    /// Panics if the file cannot be opened or written to.
    pub fn to_json(self, path: &str) -> std::io::Result<()> {
        self.parameters.to_json(path)
    }

    /// Import a Heuristics instance from a `.json`-file at the given `path`
    /// # Errors
    /// Returns an error if the file cannot be opened or read.
    /// # Panics
    /// Panics if the file cannot be read.
    pub fn from_json(path: &str) -> Result<Self, String> {
        let parameters = Parameters::from_json(path)?;
        Ok(Self::new(parameters))
    }

    #[must_use]
    pub fn get_rollout_policy_value(&mut self, game: &Game, mv: Move) -> f64 {
        self.get_move_estimation(game, mv) // todo: maybe add some randomness here
    }

    #[must_use]
    pub fn select_rollout_move(&mut self, game: &Game, moves: Vec<Move>) -> Option<Move> {
        moves
            .into_iter()
            .ord_subset_max_by_key(|mv| self.get_move_estimation(game, *mv))
    }

    #[must_use]
    pub const fn exploration_bias(&self, turn: usize) -> f64 {
        self.parameters.exploration_variables[turn - 1]
    }

    #[must_use]
    /// Recieve a value if the move would expend a special piece
    pub const fn special_use(&self, turn: usize, mv: Move) -> f64 {
        if let Move::Place(placement) = mv {
            if turn < 7 && Piece::is_special(placement.piece) {
                return self.parameters.special_cost[turn - 1];
            }
        }
        0.0
    }

    #[must_use]
    /// Recieve a value if the move would connect to an exit
    fn piece_connects_to_exit(&self, turn: usize, board: &Board, mv: Move) -> f64 {
        if let Move::Place(placement) = mv {
            if board.piece_connects_to_exit(placement) {
                return self.parameters.piece_connects_to_exit[turn - 1];
            }
        }
        0.0
    }

    #[must_use]
    /// Recieve a value if the move would connect to more than one other piece
    fn piece_connects_to_other_piece(&self, turn: usize, board: &Board, mv: Move) -> f64 {
        if let Move::Place(placement) = mv {
            let connections = board.piece_count_connections(placement);
            if connections > 1 {
                return self.parameters.piece_connects_to_other_piece[turn - 1];
            }
        }
        0.0
    }

    #[must_use]
    /// Recieve a value if the move would lock out another piece
    fn piece_locks_out_other_piece(&self, turn: usize, board: &Board, mv: Move) -> f64 {
        if let Move::Place(placement) = mv {
            let locks_out = board.piece_locks_out_other_piece(placement);
            if locks_out {
                return self.parameters.piece_locks_out_other_piece[turn - 1];
            }
        }
        0.0
    }

    #[must_use]
    /// Recieve a value for a move that is 2nd order neighbor to another piece
    fn piece_is_2nd_order_neighbor(&self, turn: usize, board: &Board, mv: Move) -> f64 {
        if let Move::Place(placement) = mv {
            let is_2nd_order_neighbor = board.piece_is_2nd_order_neighbor(placement);
            if is_2nd_order_neighbor {
                return self.parameters.piece_is_2nd_order_neighbor[turn - 1];
            }
        }
        0.0
    }

    #[must_use]
    /// Recieve a value for a move that is 3rd order neighbor to another piece
    fn piece_is_3rd_order_neighbor(&self, turn: usize, board: &Board, mv: Move) -> f64 {
        if let Move::Place(placement) = mv {
            let is_3rd_order_neighbor = board.piece_is_3rd_order_neighbor(placement);
            if is_3rd_order_neighbor {
                return self.parameters.piece_is_3rd_order_neighbor[turn - 1];
            }
        }
        0.0
    }

    pub fn update(&mut self, _turn: u8, _mv: Move, _score: f64) {}

    // #[must_use]
    // Recieve a value if the move connects to the edge of the longest network path
    // fn piece_connects_to_longest_path(&self, game: &Game, mv: Move) -> f64 {
    //     if let Move::Place(placement) = mv {
    //         let connects_to_longest_path = game.board.piece_connects_to_longest_path(placement);
    //         if connects_to_longest_path {
    //             1.0
    //         } else {
    //             0.0
    //         }
    //     } else {
    //         0.0
    //     }
    // }

    // #[must_use]
    // Recieve a value if the move would connect to the longest network path, but then ends it
    // fn piece_ends_longest_path(&self, game: &Game, mv: Move) -> f64 {
    //     if let Move::Place(placement) = mv {
    //         let ends_longest_path = game.board.piece_ends_longest_path(placement);
    //         if ends_longest_path {
    //             -1.0
    //         } else {
    //             0.0
    //         }
    //     } else {
    //         0.0
    //     }
    // }
    #[must_use]
    ///
    /// # Panics
    /// Panics if backend device can not be found to load tensor data into.
    pub fn get_move_estimation(&mut self, game: &Game, mv: Move) -> f64 {
        let board = &game.board;
        if let Some(model) = &self.move_nn {
            let device = model.devices().first().unwrap().clone();

            let board_features = DataItem::get_features(board, mv);
            let board_features = Tensor::from_data([board_features], &device);

            let meta_features = DataItem::get_heuristics(board, mv);
            let meta_features = Tensor::from_data([meta_features], &device);

            f64::from(model.forward(board_features, meta_features).into_scalar())
        } else {
            let turn = game.turn as usize;
            self.special_use(turn, mv)
                + self.piece_connects_to_exit(turn, board, mv)
                + self.piece_connects_to_other_piece(turn, board, mv)
                + self.piece_locks_out_other_piece(turn, board, mv)
                + self.piece_is_2nd_order_neighbor(turn, board, mv)
                + self.piece_is_3rd_order_neighbor(turn, board, mv)
        }
    }

    #[must_use]
    /// # Panics
    /// Panics if:
    /// * A model is not part of this instance of Heuristics.
    /// * A device can not be found to load tensor data into.
    /// * The output can not be converted to a vector.
    pub fn get_move_estimations(&mut self, game: &Game, moves: &[Move]) -> Vec<f32> {
        assert!(self.move_nn.is_some());
        let device = self.move_nn.devices().first().unwrap().clone();

        let board_features = DataItem::get_board_features(&game.board);
        let board_features: Tensor<Wgpu, 3> = Tensor::from_data(board_features, &device);
        let board_features = board_features.expand([moves.len(), 7, 7, 12]);

        let meta_features = moves
            .iter()
            .map(|mv| DataItem::get_heuristics(&game.board, *mv))
            .map(|ft| TensorData::from(ft).convert::<f32>())
            .map(|data| Tensor::<Wgpu, 1>::from_data(data, &device))
            .collect::<Vec<_>>();

        let meta_features = Tensor::stack(meta_features, 0);

        let model = self.move_nn.as_ref().unwrap();

        let output = model.forward(board_features, meta_features);

        output
            .into_data()
            .into_vec()
            .expect("Could not get move estimations")
    }

    #[must_use]
    pub fn get_exploration_value_given_heuristic(
        &self,
        mv: Move,
        mean_score: f64,
        visits: u64,
        parent_visits: u64,
        game: &Game,
        estimated_value: f64,
    ) -> f64 {
        let turn = usize::from(game.turn);
        if turn == 7 {
            return mean_score;
        }

        let ucb = mean_score;
        let exploration_bias = self.exploration_bias(turn);
        let exploration: f64 = if visits == 0 {
            // self.get_rollout_policy_value(game, mv)
            self.parameters.unexplored_value[turn]
        } else {
            Score::sqrt(Score::ln(parent_visits as f64 / visits as f64))
        };

        let exploration_term = exploration_bias * exploration;

        let exploration_term = exploration_term + self.special_use(turn, mv);

        let k = 1.;
        let n = visits as f64;
        let beta = (k / 3.0f64.mul_add(n, k)).sqrt();
        let q = (1.0 - beta).mul_add(ucb, beta * estimated_value);

        q + exploration_term
    }

    #[must_use]
    pub fn get_exploration_value(
        &mut self,
        mv: Move,
        mean_score: f64,
        visits: u64,
        parent_visits: u64,
        game: &Game,
    ) -> f64 {
        let turn = usize::from(game.turn);
        if turn == 7 {
            return mean_score;
        }

        let ucb = mean_score;
        let exploration_bias = self.exploration_bias(turn);
        let exploration: f64 = if visits == 0 {
            // self.get_rollout_policy_value(game, mv)
            self.parameters.unexplored_value[turn]
        } else {
            Score::sqrt(Score::ln(parent_visits as f64 / visits as f64))
        };

        let exploration_term = exploration_bias * exploration;

        let exploration_term = exploration_term + self.special_use(turn, mv);

        if self.move_nn.is_some() {
            unimplemented!("Neural net not yet implemented!");
        } else {
            let estimated_value = self.get_move_estimation(game, mv);
            let k = 1.;
            let n = visits as f64;
            let beta = (k / 3.0f64.mul_add(n, k)).sqrt();
            let q = (1.0 - beta).mul_add(ucb, beta * estimated_value);

            q + exploration_term
        }
    }
}

impl Default for Heuristics {
    fn default() -> Self {
        Self::from_json("./src/mcts/heuristics/default.json").unwrap()
    }
}
