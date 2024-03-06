use crate::board::Board;
use crate::game::mv::Move;
use crate::game::Game;
use crate::mcts::Score;
use crate::pieces::Piece;

use std::fs::File;
use std::io::prelude::*;
pub mod nn;
use nn::edge_strategy::EdgeStrategy as NeuralNetwork;
// use nn::face_strategy::FaceStrategy;
use ord_subset::OrdSubsetIterExt;
mod rave;

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
    pub model: String,
    pub rollout_depth: u16,
}

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

    #[must_use]
    /// Load the heuristic parameters from a json file.
    ///
    /// # Panics
    /// Panics if the file cannot be read.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened.
    ///
    pub fn from_json(path: &str) -> Result<Self, String> {
        if let Ok(mut file) = File::open(path) {
            let mut contents = String::new();

            file.read_to_string(&mut contents)
                .expect("Error loading Heuristics: Could not read file to string");

            match serde_json::from_str(&contents) {
                Ok(parameters) => Ok(parameters),
                Err(e) => Err(format!("Error loading Heuristics: {}", e)),
            }
        } else {
            Err("Error loading Heuristics: Could not find path".to_string())
        }
    }

    #[must_use]
    pub fn as_array(&self) -> [[f64; 7]; 8] {
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
            prune_minimum_node_count: 80,
            prune_alpha: 16.0,
            model: String::from("model-2"),
            rollout_depth: 5,
        }
    }
}

#[derive(Clone)]
pub struct Heuristics {
    pub parameters: Parameters,
    pub rave: Option<rave::Rave>,
    pub tree_reuse: bool,
    pub move_nn: Option<NeuralNetwork<f32>>,
}

impl Heuristics {
    #[must_use]
    pub fn new(parameters: Parameters) -> Self {
        // let mut rave = rave::Rave::new();
        // let rave = Some(rave);
        let move_nn = Some(NeuralNetwork::load(&parameters.model.clone()));

        Self {
            parameters,
            move_nn,
            rave: None,
            tree_reuse: true,
        }
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
    pub fn get_rollout_policy_value(&self, game: &Game, mv: Move) -> f64 {
        self.get_move_estimation(game, mv) // todo: maybe add some randomness here
    }

    #[must_use]
    pub fn select_rollout_move(&self, game: &Game, moves: Vec<Move>) -> Option<Move> {
        moves
            .into_iter()
            .ord_subset_max_by_key(|mv| self.get_move_estimation(game, *mv))
    }

    #[must_use]
    pub fn exploration_bias(&self, turn: usize) -> f64 {
        self.parameters.exploration_variables[turn - 1]
    }

    #[must_use]
    /// Recieve a value if the move would expend a special piece
    pub fn special_use(&self, turn: usize, mv: Move) -> f64 {
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

    pub fn update(&mut self, turn: u8, mv: Move, score: f64) {
        if let Some(rave) = &mut self.rave {
            rave.update_rave(turn, mv, score);
        }
    }

    // #[must_use]
    /// Recieve a value if the move connects to the edge of the longest network path
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
    pub fn get_move_estimation(&self, game: &Game, mv: Move) -> f64 {
        let board = &game.board;
        if let Some(nn) = &self.move_nn {
            f64::from(nn.predict(board, &mv))
        // } else if let Some(rave) = &self.rave {
        //     rave.get_move_estimation(board, mv)
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
    pub fn get_exploration_value(
        &self,
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

        if let Some(nn) = &self.move_nn {
            let k = 5.; // Increase k to weigh the neural network more heavily

            let predicted_value = f64::from(nn.predict(&game.board, &mv));
            let n = visits as f64;
            let beta = (k / 3.0f64.mul_add(n, k)).sqrt();
            let q = (1.0 - beta).mul_add(ucb, beta * predicted_value);
            q + exploration_term
        } else if let Some(rave) = self.rave.as_ref() {
            let k = 1.;
            let rave_value = rave.get_rave(turn as u8, mv);
            let rave_value = rave_value + rave.rave_exploration_bias;
            let n = visits as f64;
            let beta = (k / 3.0f64.mul_add(n, k)).sqrt();
            let q = (1.0 - beta).mul_add(ucb, beta * rave_value);

            q + exploration_term
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
        Self::from_json("./config/heuristics.json").expect("Could not load default heuristics")
    }
}
