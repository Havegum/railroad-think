#![warn(
    clippy::all,
    // clippy::restriction,
    clippy::pedantic,
    // clippy::nursery,
    clippy::cargo
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::multiple_crate_versions
)]

pub mod board;
pub mod controllers;
pub mod game;
mod identity_hasher;
pub mod mcts;
pub mod pieces;
pub mod utils;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
