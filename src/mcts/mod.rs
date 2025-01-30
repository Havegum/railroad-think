use crate::board::placement::Placement;
use crate::game::{mv::Move, roll::Roll, Game};
use rand::{RngCore, SeedableRng};

use rand_xoshiro::SplitMix64;
use std::convert::TryInto;
pub mod heuristics;
pub mod trainer;
use heuristics::Heuristics;

use rand;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{ser::SerializeMap, Serialize, Serializer};
use std::collections::HashMap;

use crate::identity_hasher::BuildHasher;

pub type Score = f64;

#[derive(Debug, Serialize, Default)]
pub struct Node {
    pub visits: u64,
    pub total_score: f64,
    pub is_terminal: bool,
    pub heuristic: f64,
    pub children: Box<[Edge]>,
}

impl Node {
    #[must_use]
    pub fn new() -> Self {
        Self {
            children: Box::new([]),
            is_terminal: false,
            total_score: 0.,
            heuristic: 0.,
            visits: 0,
        }
    }

    // Never inline, to make CPU profiling easier
    #[inline(never)]
    /// Expand the list of children to this node, but don't visit
    pub fn generate_children(&mut self, game: &mut Game) {
        self.children = game
            .generate_moves()
            .into_iter()
            .map(Edge::new)
            .collect::<Vec<Edge>>()
            .into_boxed_slice();
    }
}

#[derive(Debug)]
pub enum SingleOrMultiple {
    Single(Node),
    Multiple(HashMap<Roll, Node, BuildHasher>),
}
use SingleOrMultiple::{Multiple, Single};

impl Serialize for SingleOrMultiple {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Single(node) => node.serialize(serializer),
            Multiple(nodes) => {
                let mut map = serializer.serialize_map(Some(nodes.len()))?;
                for (k, v) in nodes {
                    map.serialize_entry(&k.to_string(), v)?;
                }
                map.end()
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Edge {
    pub mv: Move,
    pub visits: u64,
    pub mean_score: Score,
    pub heuristic_value: Option<Score>,
    pub child: Option<SingleOrMultiple>,
    pub pruned: bool,
}

impl Edge {
    #[must_use]
    pub fn new(mv: Move) -> Self {
        Self {
            mv,
            child: None,
            visits: 0,
            mean_score: 0.,
            heuristic_value: None,
            pruned: false,
        }
    }

    // Never inline, to make CPU profiling easier
    #[inline(never)]
    /// One iteration of mcts
    /// Recursively `select`s through the tree,
    /// updating the `visits` count and scores along the way
    /// Rolls are simulated every time `Move::Roll` is selected
    ///
    /// # Panics
    /// Panics if no legal moves could be selected from game position
    pub fn select(
        &mut self,
        mut game: Game,
        heuristics: &mut Heuristics,
        rng: &mut dyn RngCore,
    ) -> Score {
        // Expand and rollout
        if self.visits == 0 {
            return self.expand(game, heuristics, rng);
        }
        let mut generate_children = self.visits == 1 || game.turn == 0;

        let node = self.child.as_mut().unwrap();
        let node = match node {
            Multiple(nodes) => {
                debug_assert_eq!(self.mv, Move::Roll);
                // If edge Move is `Roll`, we don't get to choose which roll
                // to search. We have to actually roll the dice.
                let roll = game.generate_roll();
                if let Some(child) = nodes.get_mut(&roll) {
                    child
                } else {
                    let child = Node::new();
                    nodes.insert(roll, child);
                    generate_children = true;
                    nodes.get_mut(&roll).unwrap()
                }
            }
            Single(node) => node,
        };
        if generate_children {
            node.generate_children(&mut game);
        }
        if node.is_terminal {
            // Increment `visits`. But don't change `self.mean`: it's the same, still
            self.visits += 1;
            node.visits += 1;
            node.total_score += self.mean_score;
            return self.mean_score;
        }

        assert_ne!(node.children.len(), 0, "No legal moves!");
        let mut best_child_node_index = 0;

        let mut best_exploration_value = Score::MIN;

        if heuristics.move_nn.is_some() {
            for (estimated_value, edge) in heuristics
                .get_move_estimations(
                    &game,
                    &node
                        .children
                        .iter()
                        .map(|edge| edge.mv)
                        .collect::<Vec<Move>>(),
                )
                .into_iter()
                .zip(node.children.iter_mut())
            {
                edge.heuristic_value = Some(estimated_value.into());
            }
        }

        for (i, edge) in node.children.iter_mut().enumerate() {
            let child_exploration_value = edge.exploration_value(self.visits, heuristics, &game);
            if child_exploration_value >= best_exploration_value {
                best_child_node_index = i;
                best_exploration_value = child_exploration_value;
            }
        }

        let child_edge = node.children.get_mut(best_child_node_index).unwrap();

        game.do_move(child_edge.mv);
        let turn = game.turn;
        let result = child_edge.select(game, heuristics, rng);

        // Backpropagate
        self.visits += 1;
        node.visits += 1;
        node.total_score += result;
        self.mean_score = node.total_score / self.visits as f64;
        if !matches!(self.mv, Move::SetRoll(..)) {
            heuristics.update(turn, self.mv, result);
        }
        result
    }

    fn exploration_value(
        &mut self,
        parent_visits: u64,
        heuristics: &mut Heuristics,
        game: &Game,
    ) -> Score {
        if self.heuristic_value.is_none() {
            self.heuristic_value = Some(heuristics.get_move_estimation(game, self.mv));
        }

        heuristics.get_exploration_value_given_heuristic(
            self.mv,
            self.mean_score,
            self.visits,
            parent_visits,
            game,
            self.heuristic_value.unwrap(),
        )
    }

    // Never inline, to make CPU profiling easier
    #[inline(never)]
    fn expand(&mut self, game: Game, heuristics: &mut Heuristics, rng: &mut dyn RngCore) -> Score {
        debug_assert!(self.child.is_none());

        if self.mv == Move::Roll {
            self.visits = 1;
            let nodes: HashMap<Roll, Node, BuildHasher> = HashMap::with_hasher(BuildHasher);
            self.child = Some(Multiple(nodes));
            let (score, _) = Self::rollout(game, heuristics, 0, rng);
            score
        } else {
            let mut child = Node::new();
            let (score, is_terminal) = Self::rollout(game, heuristics, 0, rng);
            self.visits = 1;
            self.mean_score = score;
            child.total_score = score;
            child.is_terminal = is_terminal;
            self.child = Some(Single(child));
            score
        }
    }

    #[inline(never)]
    /// Does random moves until `game.ended`
    /// Returns `(score, depth_zero_is_terminal)`
    ///
    /// ### Ideas:
    /// * Use heuristics instead of random moves
    /// * Drop rollouts and just use the heuristic to estimate the score
    fn rollout(
        mut game: Game,
        heuristics: &mut Heuristics,
        depth: u16,
        rng: &mut dyn RngCore,
    ) -> (Score, bool) {
        if game.ended {
            return (f64::from(game.board.score()), depth == 0);
        }

        let mv = game
            .generate_moves()
            .choose(rng)
            .copied()
            .expect("Rollout failed to find a valid move");

        game.do_move(mv);
        let turn = game.turn;
        let (score, is_terminal) = Self::rollout(game, heuristics, depth + 1, rng);

        heuristics.update(turn, mv, score);

        (score, is_terminal)
    }
}

impl Default for Edge {
    fn default() -> Self {
        Self::new(Move::Place(Placement::default()))
    }
}

pub struct MonteCarloTree {
    game: Game,
    pub root: Edge,
    pub heuristics: Heuristics,
    seed: [u8; 8],
}

impl MonteCarloTree {
    #[must_use]
    pub fn new(game: Game) -> Self {
        let root = Edge::default();
        let heuristics = Heuristics::default();
        let seed: [u8; 8] = rand::thread_rng().gen();

        Self {
            game,
            root,
            heuristics,
            seed,
        }
    }

    #[must_use]
    pub fn new_from_seed(game: Game, seed: [u8; 8]) -> Self {
        let root: Edge = Edge::default();
        let heuristics = Heuristics::default();

        Self {
            game,
            root,
            heuristics,
            seed,
        }
    }

    #[must_use]
    pub fn new_with_heuristics(game: Game, heuristics: Heuristics) -> Self {
        let seed: [u8; 8] = rand::thread_rng().gen();
        Self {
            game,
            heuristics,
            root: Edge::default(),
            seed,
        }
    }

    #[must_use]
    /// # Panics
    /// Panics if most visited node is None
    pub fn calculate_depth(&self) -> u16 {
        let mut depth = 0;
        let mut edge = &self.root;
        while let Some(child) = edge.child.as_ref() {
            match child {
                Single(node) => {
                    if node.children.len() == 0 {
                        break;
                    }
                    edge = node
                        .children
                        .iter()
                        .max_by_key(|edge| edge.visits)
                        .expect("Could not find most visited node edge");
                    depth += 1;
                }
                Multiple(nodes) => {
                    let node = nodes.values().max_by_key(|node| node.visits);

                    if let Some(node) = node {
                        if node.children.len() == 0 {
                            break;
                        }
                        edge = node.children.iter().max_by_key(|edge| edge.visits).unwrap();
                        depth += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        depth
    }

    /// # Panics
    /// This function panics if the move is not possible in this state of the game.
    pub fn progress(mut mcts: Self, mv: Move, game: &mut Game) -> Self {
        game.do_move(mv);
        if !mcts.heuristics.tree_reuse {
            return Self::new_with_heuristics(game.clone(), mcts.heuristics);
        }

        if matches!(mv, Move::Roll) {
            mcts.game = game.clone();
        } else {
            mcts.game.do_move(mv);
        }

        let move_to_match = if matches!(mv, Move::SetRoll(..)) {
            Move::Roll
        } else {
            mv
        };

        mcts.root = match mcts.root.child.as_mut() {
            None => Edge::default(),
            Some(child) => match child {
                Multiple(_) => {
                    panic!("Could not `progress`. Root edge has child of type Some(Multiple)")
                }
                Single(node) => {
                    // `node` is where we're choosing some action from
                    let children: Box<[Edge]> = std::mem::take(&mut node.children);
                    let child_index = children.iter().position(|child| child.mv == move_to_match);
                    match child_index {
                        None => Edge::default(),
                        Some(index) => {
                            let mut next = Vec::from(children).swap_remove(index);
                            match next.child.as_mut() {
                                None => Edge::default(),
                                Some(child) => match child {
                                    Single(_) => next,
                                    Multiple(nodes) => {
                                        let roll = match mv {
                                        Move::SetRoll(roll) => roll,
                                        Move::Roll => {
                                          let roll = game
                                            .to_place
                                            .clone()
                                            .try_into()
                                            .unwrap_or_else(|_| panic!("Could not progress. Received Move::Roll, and could not infer roll from game due to incorrect `to_place` length."));
                                          Roll::new(roll)
                                        },
                                        _ => panic!("MCTS cannot progress to a Some(Multiple)-child unless the move is a knowable Roll. Received {:?}", &mv),
                                      };

                                        match nodes.remove(&roll) {
                                            None => Edge::default(),
                                            Some(node) => Edge {
                                                mv: Move::SetRoll(roll),
                                                visits: node.visits,
                                                mean_score: node.visits as f64 / node.total_score,
                                                child: Some(Single(node)),
                                                pruned: false,
                                                heuristic_value: None,
                                            },
                                        }
                                    }
                                },
                            }
                        }
                    }
                }
            },
        };

        mcts
    }

    /// Run one iteration of MCTS.
    /// TODO: Implement "move stack" for game that can be used to
    /// undo history and avoid cloning the board every iteration.
    /// Validate that strategy with criterion. If it's not actually
    /// faster, maybe just revert.
    /// Although, having undo-capabilities would be nice for frontend ...
    pub fn search(&mut self) {
        self.seed = SplitMix64::from_seed(self.seed).gen();
        let mut rng = SplitMix64::from_seed(self.seed);

        self.root
            .select(self.game.clone(), &mut self.heuristics, &mut rng);
    }

    pub fn search_iterations(&mut self, iterations: u64) -> &mut Self {
        for _ in 0..iterations {
            self.search();
        }
        self
    }

    pub fn search_duration(&mut self, milliseconds: u128) -> &mut Self {
        let start = std::time::Instant::now();
        while start.elapsed().as_millis() < milliseconds {
            self.search();
        }
        self
    }

    /// Return the best move given the current state of search
    /// # Panics
    /// Panics if no move could be selected from the current game position.
    #[must_use]
    pub fn best_move(&mut self) -> Move {
        match self.root.child.as_ref() {
            None => *self
                .game
                .generate_moves()
                .choose(&mut SplitMix64::from_seed(self.seed))
                .expect("Found no moves"),
            Some(child) => match child {
                Multiple(_) => Move::Roll,
                Single(node) => {
                    if let Some(child) = node
                        .children
                        .iter()
                        .filter(|edge| edge.visits != 0)
                        .max_by_key(|edge| edge.visits)
                    {
                        child.mv
                    } else if let Some(mv) = self
                        .game
                        .generate_moves()
                        .choose(&mut SplitMix64::from_seed(self.seed))
                    {
                        *mv
                    } else {
                        Move::End
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use super::*;

    #[test]
    #[allow(unused_must_use)]
    fn test_single_iteration_search() {
        let game = Game::new();
        let mut tree = MonteCarloTree::new(game);
        tree.search();
        tree.best_move();
    }

    #[test]
    fn test_bugged_board() {
        let game_seed = [167, 58, 224, 133, 94, 224, 76, 115];
        let mcts_seed = [75, 110, 21, 180, 122, 69, 56, 3];

        let game = Game::new_from_seed(game_seed);
        let mut tree = MonteCarloTree::new_from_seed(game, mcts_seed);
        tree.search_duration(1000);
    }

    #[test]
    fn test_many_iteration_search() {
        let game = Game::new();
        let mut mcts = MonteCarloTree::new(game);
        mcts.search_iterations(100);
    }

    #[test]
    fn play_random_games_test() {
        for i in 0u64..10 {
            let seed = i.to_be_bytes();
            let mut game = Game::new_from_seed(seed);
            let mut mcts = MonteCarloTree::new_from_seed(game.clone(), seed);
            while !game.ended {
                mcts.search_iterations(100);
                let mv = mcts.best_move();

                assert_eq!(mv, Move::from_str(&mv.to_string()).unwrap());
                assert_eq!(game, Game::decode(&game.encode()).unwrap());

                mcts = MonteCarloTree::progress(mcts, mv, &mut game);
            }
        }
    }

    #[test]
    fn test_play_full_game() {
        let mut game = Game::new();
        let mut mcts = MonteCarloTree::new(game.clone());
        while !game.ended {
            mcts.search_iterations(100);
            let mv = mcts.best_move();
            mcts = MonteCarloTree::progress(mcts, mv, &mut game);
        }

        println!("{}", game.encode());
        println!("MCTS SCORE {}", game.board.score());
    }

    #[test]
    fn test_play_full_game_duration() {
        let mut game = Game::new();
        let mut mcts = MonteCarloTree::new(game.clone());
        while !game.ended {
            mcts.search_duration(200);
            let mv = mcts.best_move();
            mcts = MonteCarloTree::progress(mcts, mv, &mut game);
        }
    }

    #[test]
    fn test_seeded_mcts_is_deterministic() {
        let seed = [0, 0, 0, 0, 0, 0, 0, 42];

        let mut game_a = Game::new_from_seed(seed);
        let mut mcts_a = MonteCarloTree::new_from_seed(game_a.clone(), seed);

        while !game_a.ended {
            mcts_a.search_iterations(0);
            let mv = mcts_a.best_move();
            mcts_a = MonteCarloTree::progress(mcts_a, mv, &mut game_a);
        }

        let mut game_b = Game::new_from_seed(seed);
        let mut mcts_b = MonteCarloTree::new_from_seed(game_b.clone(), seed);

        while !game_b.ended {
            mcts_b.search_iterations(0);
            let mv = mcts_b.best_move();
            mcts_b = MonteCarloTree::progress(mcts_b, mv, &mut game_b);
        }

        assert_eq!(game_a, game_b);

        // I don't think we need to prove the negative case.
        // let mut game_c = Game::new_from_seed(seed);
        // let seed = [1; 8];
        // let mut mcts_c = MonteCarloTree::new_from_seed(game_c.clone(), seed);

        // while !game_c.ended {
        //     mcts_c.search_iterations(10);
        //     let mv = mcts_c.best_move();
        //     mcts_c = MonteCarloTree::progress(mcts_c, mv, &mut game_c);
        // }

        // assert_ne!(game_a, game_c);
    }
}
