use super::board::placement::Placement;
use crate::board::Board;
use crate::pieces::Piece;
use rand_xoshiro::SplitMix64;
use serde::Serialize;
use serde_with::serde_as;

pub mod mv;
use mv::Move;

pub mod roll;
use roll::Roll;

use crate::console_log;

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]

/// Representation of the state of the game
/// * `turn` is the turn the game is in.
/// * `to_place` is a vector of u8's – the ID of each piece to place. If this vector is empty, we're at the end of a round.
/// * `expended_specials` is [Option<u8>;3] – all the specials used in the game.
/// * `board` is the game Board.
#[serde_as]
#[derive(Serialize, Clone, Debug)]
pub struct Game {
    pub turn: u8,
    pub ended: bool,
    #[serde(rename = "toPlace")]
    pub to_place: Vec<u8>,
    #[serde(rename = "expendedSpecials")]
    #[serde(serialize_with = "<[_]>::serialize")]
    pub expended_specials: [Option<u8>; 3],
    #[serde(rename = "specialPlaced")]
    pub special_placed: Option<u8>,
    pub board: Board,
    available_moves: Option<Vec<Move>>,
    #[serde(skip)]
    rng: SplitMix64,
}

impl Game {
    #[must_use]
    pub fn new() -> Self {
        let mut new = Self::default();
        new.roll();
        new
    }

    #[must_use]
    pub fn new_from_seed(seed: [u8; 8]) -> Self {
        let mut game = Self {
            rng: SplitMix64::from_seed(seed),
            ..Default::default()
        };
        game.roll();
        game
    }

    fn can_play_specials(&self) -> bool {
        self.special_placed.is_none()
            && self
                .expended_specials
                .iter()
                .filter(|x| x.is_some())
                .count()
                < 3
    }

    /// Place a piece on a square
    /// # Errors
    /// Returns Error if the piece is not playable in the current state
    pub fn place(&mut self, placement: Placement) -> Result<u8, String> {
        if 0x09 < placement.piece && placement.piece <= 0x0f {
            // Special tile
            if self.can_play_specials() {
                self.board.place(placement);
                self.special_placed = Some(placement.piece);
                for i in 0..3 {
                    if self.expended_specials[i].is_none() {
                        self.expended_specials[i] = Some(placement.piece);
                        break;
                    }
                }
                return Ok(placement.piece);
            }

            return Err(String::from(
                "Cannot play special piece. Either already placed, or three already expended",
            ));
        }

        for (i, piece) in self.to_place.iter().enumerate() {
            if piece == &placement.piece {
                self.board.place(placement);
                self.to_place.swap_remove(i);
                return Ok(placement.square.raw);
            }
        }
        Err(String::from("This piece is not playable!"))
    }

    #[must_use]
    /// # Panics
    /// Should not panic ... I just haven't gotten around to ensure the code itself assert that
    pub fn generate_roll(&mut self) -> Roll {
        static COMMON: [u8; 6] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        static TRANSITIONAL: [u8; 3] = [0x07, 0x08, 0x09];
        // let mut rng = thread_rng();

        let mut roll = [
            *COMMON.choose(&mut self.rng).unwrap(),
            *COMMON.choose(&mut self.rng).unwrap(),
            *COMMON.choose(&mut self.rng).unwrap(),
            *TRANSITIONAL.choose(&mut self.rng).unwrap(),
        ];
        roll.sort_unstable();
        Roll(roll)
    }

    /// Roll the dice and set the game state to the new roll
    /// # Panics
    /// Panics if there are pieces left to place
    pub fn roll(&mut self) -> &Vec<u8> {
        if self.generate_moves().iter().any(|mv| match mv {
            Move::Place(placement) => !Piece::is_optional(placement.piece),
            _ => false,
        }) {
            panic!("Cannot roll when there are pieces to place");
        }
        let roll = self.generate_roll();
        self.set_roll(roll);
        &self.to_place
    }

    pub fn set_roll(&mut self, roll: Roll) {
        self.turn += 1;
        self.special_placed = None;
        self.to_place = roll.to_vec();
    }

    /// Generate all possible placements given the current game state
    /// If there are no moves for the remaining rolled pieces, `Move::Roll` is added to the list.
    /// If we can play specials in this turn, specials are added to the list
    #[must_use]
    pub fn generate_moves(&mut self) -> Vec<Move> {
        if let Some(moves) = &self.available_moves {
            return moves.clone();
        }

        if self.ended {
            return vec![];
        }

        if self.turn == 0 {
            return vec![Move::Roll];
        }

        let mut moves: Vec<Move> = self
            .to_place
            .iter()
            .flat_map(|&piece| self.board.find_possible(piece))
            .map(Move::Place)
            .collect();

        if moves.is_empty() {
            moves.push(if self.turn < 7 { Move::Roll } else { Move::End });
        }

        if self.can_play_specials() {
            moves.extend(
                (0x0a..=0x0f)
                    .filter(|piece| !self.expended_specials.contains(&Some(*piece)))
                    .flat_map(|piece| self.board.find_possible(piece))
                    .map(Move::Place),
            );
        }

        self.available_moves = Some(moves.clone());

        moves
    }

    /// Play a move
    /// TODO: Change return value to Result
    pub fn do_move(&mut self, mv: Move) -> Option<String> {
        let result = match mv {
            Move::Place(placement) => self.place(placement).err(),
            Move::SetRoll(roll) => {
                if self.turn >= 7 {
                    Some(String::from("All rounds have been played!"))
                } else {
                    self.set_roll(roll);
                    None
                }
            }
            Move::Roll => {
                if self.turn >= 7 {
                    Some(String::from("All rounds have been played!"))
                } else {
                    self.roll();
                    None
                }
            }
            Move::End => {
                self.ended = true;
                None
            }
        };
        self.available_moves = None;
        result
    }

    /// Decode a string of the complete game state.
    /// Game state needs: current turn, pieces left to place, expended specials, board
    /// format: (`turn` [0-f])|(`to_place` [0-f]{2}){0,4}|(`expended_specials` [0-f]{2}){0,3})|(`specials_used_this_round` [0-f]{2}|(`board` [board])
    /// Should probably fix these:
    /// # Panics
    /// Panics if any components of the encoded string could not decode meaningfully.
    /// # Errors
    /// Errors if any components of the encoded string could not decode meaningfully.
    pub fn decode(string: &str) -> Result<Self, String> {
        let mut components = string.split('|').collect::<Vec<&str>>().into_iter();

        let turn = components.next().unwrap();
        let turn = u8::from_str_radix(turn, 16).unwrap();

        let to_place = components.next().unwrap();
        let to_place = to_place
            .chars()
            .collect::<Vec<char>>()
            .as_slice()
            .chunks(2)
            .filter_map(|p| u8::from_str_radix(p.iter().collect::<String>().as_str(), 16).ok())
            .collect::<Vec<u8>>();

        let expended_specials = components.next().unwrap();
        let expended_specials = expended_specials
            .chars()
            .collect::<Vec<char>>()
            .as_slice()
            .chunks(2)
            .filter_map(|p| u8::from_str_radix(p.iter().collect::<String>().as_str(), 16).ok())
            .collect::<Vec<u8>>();

        let expended_specials = {
            if expended_specials.len() > 3 {
                return Err(format!(
                    "Game decoding failed. Too many expended special pieces: {}.",
                    expended_specials.len()
                ));
            }
            let mut arr = [None; 3];
            for (i, special) in expended_specials.into_iter().enumerate() {
                arr[i] = Some(special);
            }
            arr
        };

        let special_placed = components.next().unwrap();
        let special_placed = u8::from_str_radix(special_placed, 16).ok();

        let board = components.next().unwrap();
        console_log!("{}", board);
        let board = Board::decode(&String::from(board));

        let rng = SplitMix64::from_seed(rand::thread_rng().gen());

        Ok(Self {
            turn,
            to_place,
            expended_specials,
            board,
            special_placed,
            ended: false,
            available_moves: None,
            rng,
        })
    }

    #[must_use]
    pub fn encode(&self) -> String {
        let turn = format!("{:01X?}", self.turn);
        let to_place = self
            .to_place
            .iter()
            .map(|piece| format!("{piece:02X?}"))
            .reduce(|cat, next| cat + &next)
            .unwrap_or_default();

        let expended_specials = self
            .expended_specials
            .iter()
            .filter_map(|&s| s)
            .map(|piece| format!("{piece:02X?}"))
            .reduce(|cat, next| cat + &next)
            .unwrap_or_default();

        let special_placed = self
            .special_placed
            .map_or_else(String::new, |piece| format!("{piece:02X?}"));

        let board = self.board.encode();

        format!("{turn}|{to_place}|{expended_specials}|{special_placed}|{board}")
    }
}

impl Default for Game {
    fn default() -> Self {
        let rng = SplitMix64::from_seed(rand::thread_rng().gen());

        Self {
            board: Board::new(),
            turn: 0,
            to_place: vec![],
            expended_specials: [None; 3],
            special_placed: None,
            ended: false,
            available_moves: None,
            rng,
        }
    }
}

impl PartialEq for Game {
    fn eq(&self, other: &Self) -> bool {
        self.turn == other.turn
            && self.to_place == other.to_place
            && self.expended_specials == other.expended_specials
            && self.board == other.board
            && self.special_placed == other.special_placed
            && self.ended == other.ended
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_game_fresh_encoding() {
        let game = Game::new();
        let encoding = game.encode();
        let decoded = Game::decode(&encoding).unwrap();
        assert_eq!(encoding, decoded.encode());
    }

    #[test]
    fn test_game_encoding() {
        let mut game = Game::new();
        let mut rng = rand::thread_rng();

        while !game.ended {
            let mv = *game.generate_moves().choose(&mut rng).unwrap();
            game.do_move(mv);

            let encoding = game.encode();
            let decoded = Game::decode(&encoding).unwrap();
            assert_eq!(encoding, decoded.encode());
        }
    }

    #[test]
    #[should_panic]
    fn test_panic_on_invalid_roll() {
        let mut game = Game::default();
        game.roll();
        game.do_move(Move::Roll);
    }

    #[test]
    fn test_game_turns() {
        let mut game = Game::new();
        let mut rng = rand::thread_rng();
        let mut current_turn = 1;
        let mut placed_pieces = 0;
        while !game.ended {
            assert_eq!(current_turn, game.turn);
            let mv = *game.generate_moves().choose(&mut rng).unwrap();
            game.do_move(mv);

            match mv {
                Move::Roll => {
                    current_turn += 1;
                    assert_eq!(4, game.to_place.len());
                }
                Move::Place(..) => {
                    placed_pieces += 1;
                }
                _ => (),
            }
        }
        assert_eq!(
            placed_pieces,
            7 * 4 + game.expended_specials.iter().flatten().count()
        );
    }

    #[test]
    fn test_seeded_game_is_deterministic() {
        let mut rng = rand::thread_rng();
        let seed = rng.gen();
        let mut game_a = Game::new_from_seed(seed);
        let mut game_b = Game::new_from_seed(seed);
        assert_eq!(game_a.to_place, game_b.to_place);

        while !game_a.ended {
            let mv = *game_a.generate_moves().choose(&mut rng).unwrap();
            game_a.do_move(mv);
            game_b.do_move(mv);
            assert_eq!(game_a.to_place, game_b.to_place);
        }
    }
}
