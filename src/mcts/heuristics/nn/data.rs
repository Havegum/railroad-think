/// See: <https://burn.dev/burn-book/basic-workflow/data.html>
use burn::data::dataset::InMemDataset;
use rusqlite::OpenFlags;

use crate::board::placement::Placement;
use crate::board::square::Square;
use crate::board::{Board, BOARD_SIZE};
use crate::game::mv::Move;
use crate::mcts::Score;
use crate::pieces::Connection;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use std::str::FromStr;

#[must_use]
/// # Panics
/// Panics if the database does not exist and then could not be created.
pub fn get_connection() -> rusqlite::Connection {
    let conn = rusqlite::Connection::open_with_flags(
        "./data.sqlite",
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    );

    if let Ok(c) = conn {
        c
    } else {
        let conn = rusqlite::Connection::open("./data.sqlite").unwrap();

        conn.execute(
            "CREATE TABLE matches (
                    id     INTEGER PRIMARY KEY,
                    board  TEXT NOT NULL,
                    move   TEXT NOT NULL,
                    score  INTEGER NOT NULL
                )",
            (), // empty list of parameters.
        )
        .unwrap();

        conn
    }
}

#[derive(Clone)]
pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct DataItem {
    board: Board,
    mv: Move,
    score: Score,
}
pub struct GameDataset {}

impl GameDataset {
    fn get_all() -> Vec<DataItem> {
        get_connection()
            .prepare("SELECT board, move, score FROM matches")
            .expect("Could not get data from sqlite database")
            .query_map([], |row| {
                Ok(DataItem {
                    board: Board::decode(row.get::<usize, String>(0).unwrap().as_str()),
                    mv: Move::from_str(row.get::<usize, String>(1).unwrap().as_str()).unwrap(),
                    score: row.get(2).unwrap(),
                })
            })
            .unwrap()
            .filter_map(Result::ok)
            .collect()
    }

    #[must_use]
    pub fn train() -> InMemDataset<DataItem> {
        let all = Self::get_all();
        let slice_len = (all.len() as f32 * 0.75).ceil() as usize;
        InMemDataset::new(all[0..slice_len].to_vec())
    }

    #[must_use]
    pub fn test() -> InMemDataset<DataItem> {
        let all = Self::get_all();
        let slice_len = (all.len() as f32 * 0.75).ceil() as usize + 1;
        InMemDataset::new(all[slice_len..].to_vec())
    }
}

impl DataItem {
    pub fn get_features(board: &Board, mv: Move) -> [[[f32; 12]; 7]; 7] {
        // let mut data = [0.0; 7 * 7 * 12];
        let mut data = [[[0.0; 12]; 7]; 7];
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let ft = board[&Square::<BOARD_SIZE>::new(x, y)]
                    .map_or([0.0; 12], Self::get_features_for_placement);
                let x = x as usize;
                let y = y as usize;
                data[y][x] = ft;
            }
        }

        if let Move::Place(placement) = mv {
            let square = placement.square;
            let x = square.x() as usize;
            let y = square.y() as usize;

            data[y][x] = Self::get_features_for_placement(placement);

            for i in 0..4 {
                data[y][x][i * 3 + 2] = 1.0;
            }
        }
        data
    }

    fn get_features_for_placement(placement: Placement) -> [f32; 12] {
        let mut cell = [0.0; 12];
        placement
            .get_networks()
            .iter()
            .flatten()
            .for_each(|network| {
                for direction in 0..4 {
                    match network[direction] {
                        Connection::Road => cell[direction * 3] = 1.0,
                        Connection::Rail => cell[direction * 3 + 1] = 1.0,
                        Connection::None => {}
                    }
                }
            });
        cell
    }

    #[must_use]
    pub fn get_heuristics(board: &Board, mv: Move) -> [f32; 7] {
        fn to_f32(boolean: bool) -> f32 {
            if boolean {
                1.0
            } else {
                0.0
            }
        }
        match mv {
            Move::Place(placement) => [
                0.0,
                1.0,
                to_f32(board.piece_connects_to_exit(placement)),
                f32::from(board.piece_count_connections(placement)) / 4.0,
                to_f32(board.piece_locks_out_other_piece(placement)),
                to_f32(board.piece_is_2nd_order_neighbor(placement)),
                to_f32(board.piece_is_3rd_order_neighbor(placement)),
            ],

            Move::Roll => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            _ => [0.0; 7],
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataBatch<B: Backend> {
    pub boards: Tensor<B, 4>,
    pub heuristics: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> Batcher<DataItem, DataBatch<B>> for DataBatcher<B> {
    fn batch(&self, items: Vec<DataItem>) -> DataBatch<B> {
        let boards = items
            .iter()
            .map(|item| DataItem::get_features(&item.board, item.mv))
            .map(|ft| TensorData::from([ft]).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 4>::from_data(data, &self.device))
            .collect();

        let heuristics = items
            .iter()
            .map(|item| DataItem::get_heuristics(&item.board, item.mv))
            .map(|ft| TensorData::from([ft]).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .collect();

        let targets = items
            .iter()
            .map(|item| item.score)
            .map(|score| TensorData::from([score]).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .collect();

        let boards = Tensor::cat(boards, 0).to_device(&self.device);
        let heuristics = Tensor::cat(heuristics, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        DataBatch {
            boards,
            heuristics,
            targets,
        }
    }
}
