use crate::identity_hasher;
use crate::identity_hasher::BuildHasher;
use std::ops::{Index, IndexMut};
use strum::IntoEnumIterator;

pub mod direction;
use direction::{
    Direction,
    Direction::{East, North, South, West},
};

// use crate::console_log;

use super::pieces::{
    get_piece, Connected, Connection,
    Connection::{Rail, Road},
};

pub mod square;
use square::Square;

pub const BOARD_SIZE: u8 = 7;

use serde::Serialize;
use serde_with::serde_as; // 1.5.1
use std::collections::{HashMap, HashSet, LinkedList};

pub mod placement;
use placement::{Orientation, Placement};

/// `Board` represents the squares and placements on a railroad ink board
/// TODO: _Optimization ideas_
/// * don't stick exits into the frontier. Check when looking up tiles instead
#[serde_as]
#[derive(Serialize, Clone, Debug)]
pub struct Board {
    #[serde(serialize_with = "<[_]>::serialize")]
    pub placements: [Option<Placement>; (BOARD_SIZE as usize).pow(2)],
    placed: Vec<u8>,
    #[serde(with = "identity_hasher::serialize")]
    pub frontier: HashMap<Square<BOARD_SIZE>, Vec<(Direction, Connection)>, BuildHasher>,
}

impl Board {
    const EXITS: [(Square<BOARD_SIZE>, (Direction, Connection)); 12] = [
        // North
        (Square { raw: 1 }, (North, Road)),
        (Square { raw: 3 }, (North, Rail)),
        (Square { raw: 5 }, (North, Road)),
        // East
        (Square { raw: 13 }, (East, Rail)),
        (Square { raw: 27 }, (East, Road)),
        (Square { raw: 41 }, (East, Rail)),
        // South
        (Square { raw: 47 }, (South, Road)),
        (Square { raw: 45 }, (South, Rail)),
        (Square { raw: 43 }, (South, Road)),
        // West
        (Square { raw: 35 }, (West, Rail)),
        (Square { raw: 21 }, (West, Road)),
        (Square { raw: 7 }, (West, Rail)),
    ];

    #[must_use]
    pub fn new() -> Self {
        let mut frontier = HashMap::with_hasher(BuildHasher);

        for (loc, connect) in Self::EXITS {
            frontier.insert(loc, vec![connect]);
        }

        Self {
            placements: [None; (BOARD_SIZE as usize).pow(2)],
            placed: vec![],
            frontier,
        }
    }

    /// Encode the board state as a hexcode string.
    /// The string is a series of 5-length chars like `3A095`
    /// * `3A` represents the square
    /// * `09` represents the piece
    /// * `5` represents the pieces orientation (rotation + flip)
    ///
    /// TODO: Check which is faster
    ///
    ///   Thesis:
    ///     using the `placed`-vec is faster on smaller boards
    ///     but it creates overhead with little speed benefit on
    ///     more filled-in boards.
    ///     If its the _only_ thing on the heap, remove this.
    ///     no noticeable difference ... maybe keep it for now though and remove if memory becomes a problem
    /// ```rs
    /// self
    ///   .placements
    ///   .iter()
    ///   .filter_map(|p| p.as_ref())
    /// ```
    #[must_use]
    pub fn encode(&self) -> String {
        self.placed
            .iter()
            .filter_map(|idx| self[idx])
            .fold(String::new(), |acc, placement| {
                acc + &format!("{placement:?}")
            })
    }

    #[inline]
    #[must_use]
    /// # Panics
    /// Panics if the encoding is bad
    pub fn decode(string: &str) -> Self {
        let mut board = Self::new();
        for chunk in string.chars().collect::<Vec<char>>().as_slice().chunks(5) {
            let x = chunk[0].to_digit(10).unwrap_or_else(|| {
                panic!(
                    "Could not decode position. Expected [0-9], received {}.",
                    chunk[0]
                )
            });
            let y = chunk[1].to_digit(17).unwrap_or_else(|| {
                panic!(
                    "Could not decode position. Expected [a-g], received {}.",
                    chunk[1]
                )
            }) - 10;
            let x = x as u8;
            let y = y as u8;

            let piece_id = chunk[2..=3]
                .iter()
                .map(|c| c.to_digit(16).unwrap() as u8)
                .sum::<u8>();

            let mut flip = false;
            let rotation = {
                let rot = chunk[4].to_digit(10).unwrap_or_else(|| {
                    panic!(
                        "Could not decode rotation. Expected [0-8], received {}.",
                        chunk[4]
                    )
                });
                if rot > 3 {
                    flip = true;
                    rot - 4
                } else {
                    rot
                }
            } as u8;

            let placement = Placement {
                square: Square {
                    raw: x + y * BOARD_SIZE,
                },
                piece: piece_id,
                orientation: Orientation { rotation, flip },
            };

            board.place(placement);
        }

        board
    }

    fn get(&self, square: Square<BOARD_SIZE>) -> Option<&Placement> {
        if square.raw < BOARD_SIZE.pow(2) {
            self[&square].as_ref()
        } else {
            None
        }
    }

    fn has(&self, square: Square<BOARD_SIZE>) -> bool {
        self.get(square).is_some()
    }

    fn iter(&self) -> impl Iterator<Item = &Placement> {
        self.placed
            .iter()
            .filter_map(move |index| self[index].as_ref())
    }

    fn values(&self) -> impl Iterator<Item = &Placement> {
        self.iter()
    }

    fn insert(&mut self, square: Square<BOARD_SIZE>, placement: Placement) -> Option<Placement> {
        self.placed.push(square.raw);
        std::mem::replace(&mut self[&square], Some(placement))
    }

    #[must_use]
    /// Find all valid locations given the current board state and a piece
    /// # Panics
    /// Panics if the `piece_id` doesn't correspond to anything
    pub fn find_possible(&self, piece_id: u8) -> Vec<Placement> {
        // self.frontier is the set of squares with an open connection
        // For each frontier square, check all permutations of the given tile
        // Return a vector of valid placements
        // (square x, square y), (piece, rotation, flip)

        let piece = get_piece(piece_id)
            .unwrap_or_else(|| panic!("Piece ID not found. Saw \"{:#04x}\"", piece_id));

        let mut valid = vec![];

        for (&square, arr) in &self.frontier {
            if self.has(square) {
                continue;
            }
            for &(direction, connection) in arr {
                for orientation in piece.get_permutations() {
                    let piece = piece.permute(orientation);

                    // piece is valid if it has a connection to the origin placement AND
                    let is_valid = piece.has_connection(direction, connection);
                    // if none of this pieces connections would conflict with existing connections
                    let is_valid = is_valid
                        && Direction::iter()
                            .map(|dir| (dir, piece.connection(dir)))
                            .filter(|(_, connection)| connection.is_some())
                            .all(|(dir, con)| {
                                self.get(Self::get_neighbor(square, dir)).map_or_else(
                                    || !{
                                        square.is_border() && {
                                            Self::EXITS.iter().any(
                                                |(exit_square, (exit_dir, exit_con))| {
                                                    exit_square == &square
                                                        && exit_dir == &dir
                                                        && match con {
                                                            Connection::None => false,
                                                            connection => &connection != exit_con,
                                                        }
                                                },
                                            )
                                        }
                                    },
                                    |place| match place.connection(dir.inverse()) {
                                        Connection::None => true,
                                        connection => connection == con,
                                    },
                                )
                            });

                    if is_valid {
                        valid.push(Placement {
                            square,
                            piece: piece_id,
                            orientation,
                        });
                    }
                }
            }
        }

        valid
    }

    #[must_use]
    pub fn piece_connects_to_exit(&self, placement: Placement) -> bool {
        if !placement.square.is_border() {
            return false;
        }

        Direction::iter()
            .map(|dir| (dir, placement.connection(dir)))
            .filter(|(_, connection)| connection.is_some())
            .any(|(dir, con)| {
                Self::EXITS
                    .iter()
                    .any(|(exit_square, (exit_dir, exit_con))| {
                        exit_square == &placement.square && exit_dir == &dir && exit_con == &con
                    })
            })
    }

    #[must_use]
    pub fn piece_count_connections(&self, placement: Placement) -> u8 {
        Direction::iter()
            .map(|dir| (dir, placement.connection(dir)))
            .filter(|(_, connection)| connection.is_some())
            .filter(|(dir, con)| {
                self.get(Self::get_neighbor(placement.square, *dir))
                    .map_or_else(|| false, |place| &place.connection(dir.inverse()) == con)
            })
            .count() as u8
    }

    #[must_use]
    pub fn piece_locks_out_other_piece(&self, placement: Placement) -> bool {
        Direction::iter()
            .filter(|dir| placement.connection(*dir).is_none())
            .any(|dir| {
                self.get(Self::get_neighbor(placement.square, dir))
                    .map_or_else(|| false, |place| place.connection(dir.inverse()).is_some())
            })
    }

    #[must_use]
    pub fn piece_is_2nd_order_neighbor(&self, placement: Placement) -> bool {
        Direction::iter()
            .filter(|dir| placement.connection(*dir).is_some())
            .map(|source_dir| Self::get_neighbor(placement.square, source_dir))
            .filter(|square| self.get(*square).is_none())
            .filter(|square| !square.out_of_bounds())
            .any(|square| self.frontier.contains_key(&square))
    }

    #[must_use]
    pub fn piece_is_3rd_order_neighbor(&self, placement: Placement) -> bool {
        Direction::iter()
            .filter(|dir| placement.connection(*dir).is_some())
            .map(|dir| (dir, Self::get_neighbor(placement.square, dir)))
            .filter(|(_, square)| self.get(*square).is_none())
            .any(|(source_dir, square)| {
                Direction::iter()
                    .filter(|dir| dir != &source_dir.inverse())
                    .map(|dir| Self::get_neighbor(square, dir))
                    .filter(|square| !square.out_of_bounds())
                    .any(|square| self.frontier.contains_key(&square))
            })
    }

    /// Add a placement to the board, and update internal state to match
    pub fn place(&mut self, placement: Placement) {
        let square = placement.square;

        // remove placement location from frontier if a connection is made
        let frontier = self.frontier.get_mut(&square);
        if let Some(frontier) = frontier {
            frontier.retain(|&(direction, connection)| {
                !placement.has_connection(direction, connection)
            });
            if frontier.is_empty() {
                self.frontier.remove(&square);
            }
        };

        // add neighbour locations to frontier:
        //   if placement has a connection that direction and
        //   if that tile does not connect with the same tile type
        for direction in Direction::iter() {
            let square = Self::get_neighbor(square, direction);
            let connection = placement.connection(direction);
            if connection.is_none() || square.out_of_bounds() {
                continue;
            }

            if let Some(placement) = self.get(square) {
                let connects = placement.has_connection(direction.inverse(), connection);

                if connects {
                    continue;
                }
            }

            self.frontier.entry(square).or_default();

            if let Some(frontier) = self.frontier.get_mut(&square) {
                frontier.push((direction.inverse(), connection));
            }
        }

        // add `placement` to the board
        self.insert(square, placement);
    }

    fn get_networks(&self) -> Vec<(Vec<Square<BOARD_SIZE>>, u8)> {
        // visited = {}
        // networks = []
        //
        // for each placed tile:
        //   if visited { continue }
        //
        //   current_network = []
        //   filter to connected exits
        //   if exit is visited, return
        //
        //   Breadth first search through connections
        //   for each node
        //     if node is an exit
        //       add exit to visited list
        //       push exit to current network
        //   networks.push((current_network, connected_exits))
        // networks
        let mut visited: HashSet<(Square<BOARD_SIZE>, u8)> = HashSet::new();
        let mut networks: Vec<(Vec<Square<BOARD_SIZE>>, u8)> = Vec::new();

        for placement in self.values() {
            for network_index in [0, 1] {
                if visited.contains(&(placement.square, network_index)) {
                    continue;
                }

                if placement.get_networks()[network_index as usize].is_none() {
                    continue;
                }

                let mut to_visit = LinkedList::from([(placement, network_index)]);
                let mut current_network: Vec<Square<BOARD_SIZE>> = Vec::new();
                let mut exits_in_current_network = 0_u8;

                while !to_visit.is_empty() {
                    let (placement, network_index) = to_visit.pop_front().unwrap();
                    if !visited.insert((placement.square, network_index)) {
                        continue;
                    }

                    current_network.push(placement.square);

                    let connections = placement.get_connections_in_network(network_index);
                    for (direction, connection) in connections {
                        if let Some(neighbor) =
                            self.get(Self::get_neighbor(placement.square, direction))
                        {
                            if let Some(neighbor_network_index) =
                                neighbor.get_connected_network(&direction.inverse(), &connection)
                            {
                                if !visited.contains(&(neighbor.square, neighbor_network_index)) {
                                    to_visit.push_back((neighbor, neighbor_network_index));
                                }
                            }
                        } else {
                            match (
                                placement.square.x(),
                                placement.square.y(),
                                direction,
                                connection,
                            ) {
                                (1 | 5, 0, North, Road)
                                | (3, 0, North, Rail)
                                | (6, 1 | 5, East, Rail)
                                | (6, 3, East, Road)
                                | (5 | 1, 6, South, Road)
                                | (3, 6, South, Rail)
                                | (0, 5 | 1, West, Rail)
                                | (0, 3, West, Road) => {
                                    exits_in_current_network += 1;
                                }
                                _ => (),
                            }
                        }
                    }
                }
                networks.push((current_network, exits_in_current_network));
            }
        }

        networks
    }

    #[must_use]
    pub fn pct_filled(&self) -> f32 {
        let total = f32::from(BOARD_SIZE.pow(2));
        let filled = self.placements.iter().filter(|p| p.is_some()).count() as f32;
        filled / total
    }

    #[must_use]
    pub fn score_center_tiles(&self) -> i32 {
        #[rustfmt::skip]
        let center_tiles: [(u8, u8); 9] = [
            (2, 2), (3, 2), (4, 2),
            (2, 3), (3, 3), (4, 3),
            (2, 4), (3, 4), (4, 4),
        ];

        center_tiles
            .iter()
            .map(|&(x, y)| Square::new(x, y))
            .filter(|&square| self.has(square))
            .count() as i32
    }

    #[must_use]
    pub fn score_open_end(&self) -> i32 {
        self.frontier
            .iter()
            .flat_map(|(square, connections)| connections.iter().map(move |(d, c)| (square, d, c)))
            .filter_map(|(&square, &dir, _)| self.get(Self::get_neighbor(square, dir)))
            .count() as i32
    }

    #[must_use]
    pub fn score_network(&self) -> i32 {
        let networks = self.get_networks();
        let network_score = networks
            .iter()
            .map(|(_, exits)| match exits {
                12 => 45,
                exits => (exits.saturating_sub(1)) * 4,
            })
            .sum::<u8>();
        i32::from(network_score)
    }

    /// Score the board.
    /// We just apply the rules of the game, but assume we always get the longest road and rail
    #[must_use]
    pub fn score(&self) -> i32 {
        let network_score = self.score_network();

        let open_end_score = self.score_open_end();
        let center_tile_score = self.score_center_tiles();

        let end_nodes = self.get_end_nodes();
        let longest_rail = self.get_longest(Rail, &end_nodes);
        let rail_score = longest_rail.len() as i32;
        let longest_road = self.get_longest(Road, &end_nodes);
        let road_score = longest_road.len() as i32;

        let score = network_score + road_score + rail_score + center_tile_score;
        score - open_end_score
    }

    /// For each node, DFS through all connected nodes of same type
    fn get_longest(
        &self,
        connection: Connection,
        end_nodes: &HashSet<Square<BOARD_SIZE>>,
    ) -> Vec<Square<BOARD_SIZE>> {
        let mut longest = None;

        for &loc in end_nodes {
            // If `loc` is not a placed tile, continue,
            // If  place has no `connection`-type connections, continue
            match self.get(loc) {
                None => continue,
                Some(place) => {
                    let any_appropriate_connections =
                        Direction::iter().any(|dir| place.has_connection(dir, connection));

                    if !any_appropriate_connections {
                        continue;
                    }
                }
            };

            let next = self.depth_first_find_longest(loc, connection, &mut HashSet::new());
            match longest {
                None => longest = Some(next),
                Some(vec) => longest = Some(if vec.len() > next.len() { vec } else { next }),
            }
        }
        longest.unwrap_or_default()
    }

    /// Get all connected nodes of same type
    ///
    /// First, find all connections to current node,
    /// Filter visited connections
    ///   If no unvisited connections, return depth,
    ///   Else
    ///     add self to visited
    ///     For each connected node,
    ///       get dept by recursively run `depth_first_find_longest`
    ///     next, remove self from visited, and
    ///     return largest
    fn depth_first_find_longest(
        &self,
        node: Square<BOARD_SIZE>,
        connection: Connection,
        visited: &mut HashSet<Square<BOARD_SIZE>>,
    ) -> Vec<Square<BOARD_SIZE>> {
        let connected = self
            .get_connected_of_type(node, connection)
            .into_iter()
            .filter(|node| !visited.contains(node))
            .collect::<Vec<_>>();
        if connected.is_empty() {
            return vec![node];
        }
        visited.insert(node);
        let mut longest = None;
        for &node in &connected {
            let next = self.depth_first_find_longest(node, connection, visited);

            match longest {
                None => longest = Some(next),
                Some(vec) => longest = Some(if vec.len() > next.len() { vec } else { next }),
            }
        }
        visited.remove(&node);
        let mut longest = longest.unwrap_or_default();
        longest.push(node);
        longest
    }

    /**
     * For each direction
     * 1. filter to include only the relevant connection type (Rail or Road)
     * 3. filter to include only if the there is some tile at that location
     * 2. map to the (x, y) tuple
     * 4. collect
     */
    fn get_connected_of_type(
        &self,
        square: Square<BOARD_SIZE>,
        connection: Connection,
    ) -> Vec<Square<BOARD_SIZE>> {
        self.get(square).map_or(vec![], |place| {
            Direction::iter()
                .filter(|&direction| place.has_connection(direction, connection))
                .map(|direction| (direction, Self::get_neighbor(square, direction)))
                .filter_map(|(direction, square)| {
                    self.get(square).and_then(|neighbor| {
                        if neighbor.has_connection(direction.inverse(), connection) {
                            Some(neighbor.square)
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
    }

    /// Find end nodes. We use this later for finding longest paths.
    /// Three ways for a tile to be an end node:
    ///  1. it has an open ending
    ///  2. it's connected to the edge of the map
    ///  3. it's a transition tile
    fn get_end_nodes(&self) -> HashSet<Square<BOARD_SIZE>> {
        // Find placements with open connections (always lay opposite to frontier)
        let mut end_nodes = self
            .frontier
            .iter()
            .flat_map(|(&from, borders)| {
                borders
                    .iter()
                    .map(move |&(direction, _)| Self::get_opposite_neighbor(from, direction))
            })
            .filter(|&square| square.out_of_bounds())
            .collect::<HashSet<Square<BOARD_SIZE>>>();

        // Look for transitional and edge connections:
        for placement in self.iter() {
            let square = placement.square;
            if end_nodes.contains(&square) {
                continue;
            }

            let is_transitional =
                matches!(placement.piece, 0x08 | 0x09 | 0x0A | 0x0B | 0x0E | 0x0F);

            let x = square.x();
            let y = square.y();

            if is_transitional
                || y == 0 && placement.has_some_connection(North)
                || x == 6 && placement.has_some_connection(West)
                || y == 6 && placement.has_some_connection(South)
                || x == 0 && placement.has_some_connection(East)
            {
                end_nodes.insert(square);
            }
        }

        end_nodes
    }
    fn get_opposite_neighbor(from: Square<BOARD_SIZE>, direction: Direction) -> Square<BOARD_SIZE> {
        let x = from.x();
        let y = from.y();
        match direction {
            North => Square::new(x, y + 1),
            East => Square::new(x.wrapping_sub(1), y),
            South => Square::new(x, y.wrapping_sub(1)),
            West => Square::new(x + 1, y),
        }
    }

    /// Get the neighbor in the given direction
    /// Note: This will wrap around at 255, NOT the board boundary.
    /// TODO: This should probably be a method on Square, and could be an impl of Add<Direction>
    /// Also, return an Option<Square> instead of wrapping around.
    fn get_neighbor(from: Square<BOARD_SIZE>, direction: Direction) -> Square<BOARD_SIZE> {
        let x = from.x();
        let y = from.y();

        match direction {
            North => Square::new(x, y.wrapping_sub(1)),
            East => Square::new(x + 1, y),
            South => Square::new(x, y + 1),
            West => Square::new(x.wrapping_sub(1), y),
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<&u8> for Board {
    type Output = Option<Placement>;
    fn index(&self, index: &u8) -> &Option<Placement> {
        &self.placements[*index as usize]
    }
}

impl IndexMut<&u8> for Board {
    fn index_mut(&mut self, index: &u8) -> &mut Option<Placement> {
        &mut self.placements[*index as usize]
    }
}

impl<const S: u8> Index<&Square<S>> for Board {
    type Output = Option<Placement>;

    fn index(&self, square: &Square<S>) -> &Option<Placement> {
        &self.placements[square.raw as usize]
    }
}

impl<const S: u8> IndexMut<&Square<S>> for Board {
    fn index_mut(&mut self, square: &Square<S>) -> &mut Option<Placement> {
        &mut self.placements[square.raw as usize]
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.placements == other.placements
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cannot_place_tiles_into_wrong_network_type() {
        let encoding = String::from(
      "6F0315F0113G0122G0102F0121F0220F0310B0311B0231C0301D0133A0303B0104B0315B0D06B0315C0305D010",
    );
        let board = Board::decode(&encoding);

        let candidates = board.find_possible(3);
        assert_eq!(candidates.len(), 0);
    }
}
