//! This module defines an abstract representation of a TalentSched instance.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalentSchedInstance {
    pub nb_scenes: usize,
    pub nb_actors: usize,
    pub cost: Vec<usize>,
    pub duration: Vec<usize>,
    pub actors: Vec<Vec<usize>>,
}
