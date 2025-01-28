#![allow(clippy::all)]
// use crate::board::Board;
use crate::game::mv::Move;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::str::FromStr;

type Turn = u8;

#[derive(Clone)]
pub struct Rave {
    pub rave_jitter: f64,
    pub rave_exploration_bias: f64,
    rave: HashMap<Move, Value>,
    local_rave: HashMap<(Turn, Move), Value>,
}

impl Rave {
    pub fn new() -> Self {
        Self::default()
    }

    /// # Errors
    /// Returns an error if the path could not be written to
    pub fn load_rave(&mut self, path: &str) {
        let mut file = File::open(path).expect("Error loading Rave: Could not find path");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Error loading Rave: Could not read file to string");

        let map: Result<HashMap<_, _, _>, ()> = contents
            .split('\n')
            .skip(1)
            .map(|row| row.split(',').collect::<Vec<_>>())
            .map(|row| {
                let place = Move::from_str(row[0])?;
                let visits = row[1].parse().expect("load_rave: could not parse visits");
                let total = row[2].parse().expect("load_rave: could not parse total");
                Ok((place, Value::new_with_data(visits, total)))
            })
            .collect();
        let map = map.expect("Error loading Rave: Could not parse file to HashMap");
        self.rave = map;
    }

    /// Export this instance of Local Rave to a `.csv`-file at the given `path`
    ///
    /// # Errors
    /// Returns an error if the path could not be written to
    pub fn dump_local_rave(&self, path: &str) -> Result<(), std::io::Error> {
        let header = String::from("turn,move,visits,total");
        let data = self
            .local_rave
            .iter()
            .map(|((turn, mv), val)| format!("{turn},{mv:?},{},{}", val.visits, val.total_score))
            .fold(header, |csv, next| format!("{csv}\n{next}"));

        let mut file = File::create(path)?;
        let mut buffer = Vec::new();
        write!(&mut buffer, "{data}")?;
        file.write_all(buffer.as_slice())
    }

    pub fn load_local_rave(&mut self, path: &str) {
        let mut file = File::open(path).expect("Error loading Rave: Could not find path");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Error loading Rave: Could not read file to string");

        let map: Result<HashMap<_, _, _>, ()> = contents
            .split('\n')
            .skip(1)
            .map(|row| row.split(',').collect::<Vec<_>>())
            .map(|row| {
                let turn = u8::from_str(row[0]).expect("load_local_rave: could not parse turn");
                let place = Move::from_str(row[1])?;
                let visits = row[2].parse().expect("load_rave: could not parse visits");
                let total = row[3].parse().expect("load_rave: could not parse total");
                Ok(((turn, place), Value::new_with_data(visits, total)))
            })
            .collect();
        let map = map.expect("Error loading Rave: Could not parse file to HashMap");
        self.local_rave = map;
    }

    /// Reset the weights of the global rapid action value estimation
    pub fn uniform_rave(&mut self, weight: u128) {
        self.rave.values_mut().for_each(|val| val.uniform(weight));
    }

    /// Reset the weights of the local rapid action value estimation
    pub fn uniform_local_rave(&mut self, weight: u128) {
        self.local_rave
            .values_mut()
            .for_each(|val| val.uniform(weight));
    }

    #[must_use]
    pub fn get_rave(&self, turn: u8, mv: Move) -> f64 {
        const MAGIC_DEFAULT_RAVE_SCORE: f64 = f64::MAX;
        match self.local_rave.get(&(turn, mv)) {
            None => MAGIC_DEFAULT_RAVE_SCORE,
            Some(rave_value) => rave_value.get_mean(),
        }
    }

    pub fn update_rave(&mut self, turn: u8, mv: Move, score: f64) {
        self.local_rave
            .entry((turn, mv))
            .or_default()
            .update(score as i32);
    }
}

impl Default for Rave {
    fn default() -> Self {
        Self {
            rave_jitter: 0.5,
            rave_exploration_bias: 18.0,
            rave: HashMap::new(),
            local_rave: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    pub visits: u128,
    pub total_score: i128,
    mean: f64,
}

impl Value {
    // #[must_use]
    // pub fn new(score: i32) -> Self {
    //     Self {
    //         visits: 1,
    //         total_score: i128::from(score),
    //         mean: f64::from(score),
    //     }
    // }
    #[must_use]
    pub fn new_with_data(visits: u128, total_score: i128) -> Self {
        Self {
            visits,
            total_score,
            mean: total_score as f64 / visits as f64,
        }
    }

    pub fn update(&mut self, score: i32) {
        self.total_score += i128::from(score);
        self.visits += 1;
        self.mean = self.total_score as f64 / self.visits as f64;
    }

    #[must_use]
    pub fn get_mean(&self) -> f64 {
        self.mean
    }

    // #[must_use]
    // pub fn get_total(&self) -> i128 {
    //     self.total_score
    // }

    // #[must_use]
    // pub fn get_visits(&self) -> u128 {
    //     self.visits
    // }
    pub fn uniform(&mut self, weight: u128) {
        self.total_score = (self.mean * weight as f64) as i128;
        self.visits = weight;
    }
}

impl Default for Value {
    fn default() -> Self {
        Self {
            visits: 0,
            total_score: 0,
            mean: 0.,
        }
    }
}
