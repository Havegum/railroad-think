use crate::{board::Board, game::mv::Move};

mod attempt_2;
pub mod edge_strategy;
// pub mod face_strategy;

trait NNHeuristic {
    fn predict(&self, board: &Board, mv: Vec<Move>) -> Vec<f32>;
}
