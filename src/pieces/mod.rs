use serde::{Deserialize, Serialize};
use std::option::Option;
use strum_macros::Display;

use super::board::direction::Direction;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, Debug)]
pub enum Connection {
    Road,
    Rail,
    None,
}

impl Connection {
    pub fn is_none(self) -> bool {
        matches!(self, Connection::None)
    }

    pub fn is_some(self) -> bool {
        !self.is_none()
    }
}

use super::board::placement::Orientation;
use Connection::{None, Rail, Road};

// IDEA:
// |  networks is a list of connections indexed
// |  by direction [North, East, South, West],
// |  and an integer representing the network the connection belongs to
// Maybe ... idk.
#[derive(Clone, Copy, Serialize, Debug)]
pub struct Piece {
    pub networks: [Option<[Connection; 4]>; 2],
    pub rotations: [bool; 4],
    pub flippable: bool,
}

impl Piece {
    pub fn get_permutations(&self) -> Vec<Orientation> {
        let mut valid_permutations = Vec::new();

        for flip in [true, false] {
            if !self.flippable && flip {
                continue;
            }
            for rotation in 0..4_u8 {
                if !self.rotations[rotation as usize] {
                    continue;
                }
                valid_permutations.push(Orientation::new(rotation, flip));
            }
        }

        valid_permutations
    }

    pub fn permute(mut self, permutation: Orientation) -> Self {
        for connections in &mut self.networks {
            if let Some(mut c) = connections.take() {
                if permutation.flip {
                    c.reverse();
                    c[..].rotate_right(1);
                };
                c[..].rotate_right(permutation.rotation as usize);

                connections.replace(c);
            }
        }
        self
    }

    pub fn is_optional(piece: u8) -> bool {
        piece >= 0x0a
    }

    pub fn is_special(piece: u8) -> bool {
        0x0A < piece && piece < 0x10
    }

    pub fn get_networks(piece: u8, orientation: Orientation) -> [Option<[Connection; 4]>; 2] {
        get_piece(piece).map_or([Option::None, Option::None], |piece| {
            piece.permute(orientation).networks
        })
    }
}

pub trait Connected {
    fn connection(&self, direction: Direction) -> Connection;
    fn has_connection(&self, direction: Direction, connection: Connection) -> bool {
        self.connection(direction) == connection
    }

    fn has_some_connection(&self, direction: Direction) -> bool {
        self.connection(direction) != Connection::None
    }
}

impl Connected for Piece {
    fn connection(&self, direction: Direction) -> Connection {
        for net in self.networks.iter().flatten() {
            let connection = net[direction as usize];
            if connection != Connection::None {
                return connection;
            }
        }
        Connection::None
    }
}

pub const fn get_piece(id: u8) -> Option<Piece> {
    match id {
        0x01 => Some(Piece {
            // L rail
            networks: [Some([Rail, Rail, None, None]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x02 => Some(Piece {
            // T rail
            networks: [Some([Rail, Rail, None, Rail]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x03 => Some(Piece {
            // I rail
            networks: [Some([Rail, None, Rail, None]), Option::None],
            rotations: [true, true, false, false],
            flippable: false,
        }),

        0x04 => Some(Piece {
            // L road
            networks: [Some([Road, Road, None, None]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x05 => Some(Piece {
            // T road
            networks: [Some([Road, Road, None, Road]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x06 => Some(Piece {
            // I road
            networks: [Some([Road, None, Road, None]), Option::None],
            rotations: [true, true, false, false],
            flippable: false,
        }),

        0x07 => Some(Piece {
            // overpass
            networks: [
                Some([Road, None, Road, None]),
                Some([None, Rail, None, Rail]),
            ],
            rotations: [true, true, false, false],
            flippable: false,
        }),

        0x08 => Some(Piece {
            // I trans
            networks: [Some([Rail, None, Road, None]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x09 => Some(Piece {
            // L trans
            networks: [Some([Road, Rail, None, None]), Option::None],
            rotations: [true, true, true, true],
            flippable: true,
        }),

        0x0A => Some(Piece {
            // X T road
            networks: [Some([Road, Road, Rail, Road]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x0B => Some(Piece {
            // X T rail
            networks: [Some([Rail, Rail, Road, Rail]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x0C => Some(Piece {
            // X road
            networks: [Some([Road, Road, Road, Road]), Option::None],
            rotations: [true, false, false, false],
            flippable: false,
        }),

        0x0D => Some(Piece {
            // X rail
            networks: [Some([Rail, Rail, Rail, Rail]), Option::None],
            rotations: [true, false, false, false],
            flippable: false,
        }),

        0x0E => Some(Piece {
            // X L
            networks: [Some([Road, Road, Rail, Rail]), Option::None],
            rotations: [true, true, true, true],
            flippable: false,
        }),

        0x0F => Some(Piece {
            // X I
            networks: [Some([Road, Rail, Road, Rail]), Option::None],
            rotations: [true, true, false, false],
            flippable: false,
        }),
        _ => Option::None,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn vanilla_piece_permutations() {
        assert_eq!(4, get_piece(0x01).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x02).unwrap().get_permutations().len());
        assert_eq!(2, get_piece(0x03).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x04).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x05).unwrap().get_permutations().len());
        assert_eq!(2, get_piece(0x06).unwrap().get_permutations().len());
        assert_eq!(2, get_piece(0x07).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x08).unwrap().get_permutations().len());
        assert_eq!(8, get_piece(0x09).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x0A).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x0B).unwrap().get_permutations().len());
        assert_eq!(1, get_piece(0x0C).unwrap().get_permutations().len());
        assert_eq!(1, get_piece(0x0D).unwrap().get_permutations().len());
        assert_eq!(4, get_piece(0x0E).unwrap().get_permutations().len());
        assert_eq!(2, get_piece(0x0F).unwrap().get_permutations().len());
    }
}
