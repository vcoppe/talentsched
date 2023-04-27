use std::collections::HashMap;

use clustering::{kmeans, Elem};
use ddo::{Compression, Problem, Decision};
use smallbitset::Set64;

use crate::instance::TalentSchedInstance;

use super::model::{TalentSched, TalentSchedState};

struct Item<'a> {
    id: usize,
    pb: &'a TalentSched,
}

impl<'a> Elem for Item<'a> {
    fn dimensions(&self) -> usize {
        self.pb.instance.nb_actors
    }

    fn at(&self, i: usize) -> f64 {
        self.pb.instance.actors[i][self.id] as f64
    }
}

pub struct TalentSchedCompression<'a> {
    pub problem: &'a TalentSched,
    pub meta_problem: TalentSched,
    pub membership: HashMap<isize, isize>,
    pub members: Vec<Set64>,
}

impl<'a> TalentSchedCompression<'a> {
    pub fn new(problem: &'a TalentSched, n_meta_scenes: usize) -> Self {
        let mut elems = vec![];
        for i in 0..problem.instance.nb_scenes {
            elems.push(Item {
                id: i,
                pb: problem,
            });
        }
        let clustering = kmeans(n_meta_scenes, Some(0), &elems, 1000);

        let actors = Self::compute_meta_actors(problem, &clustering.membership, n_meta_scenes);
        
        let meta_instance = TalentSchedInstance {
            nb_scenes: problem.instance.nb_scenes,
            nb_actors: problem.instance.nb_actors,
            cost: problem.instance.cost.clone(),
            duration: problem.instance.duration.clone(),
            actors,
        };

        let meta_problem = TalentSched::new(meta_instance);

        let mut members = vec![Set64::default(); n_meta_scenes];
        for (i, j) in clustering.membership.iter().copied().enumerate() {
            members[j].add_inplace(i);
        }

        let mut membership = HashMap::new();
        for i in 0..n_meta_scenes {
            for k in members[i].iter() {
                for l in members[i].iter() {
                    membership.insert(k as isize, l as isize);
                }
            }
        }

        TalentSchedCompression {
            problem,
            meta_problem,
            membership,
            members,
        }
    }

    fn compute_meta_actors(pb: &TalentSched, membership: &Vec<usize>, n_meta_scenes: usize) -> Vec<Vec<usize>> {
        let mut meta_actors = vec![vec![1; n_meta_scenes]; pb.instance.nb_actors];

        for (i, j) in membership.iter().copied().enumerate() {
            for actor in 0..pb.instance.nb_actors {
                meta_actors[actor][j] &= pb.instance.actors[actor][i];
            }
        }
        
        let mut meta_actors_expanded = vec![vec![]; pb.instance.nb_actors];
        for j in membership.iter().copied() {
            for actor in 0..pb.instance.nb_actors {
                meta_actors_expanded[actor].push(meta_actors[actor][j]);
            }
        }

        meta_actors_expanded
    }
}

impl<'a> Compression for TalentSchedCompression<'a> {
    type State = TalentSchedState;

    fn get_compressed_problem(&self) -> &dyn Problem<State = Self::State> {
        &self.meta_problem
    }

    fn compress(&self, state: &TalentSchedState) -> TalentSchedState {
        if !state.maybe_scenes.is_empty() {
            TalentSchedState {
                scenes: Set64::default(),
                maybe_scenes: Set64::default(),
            }
        } else {
            state.clone()
        }
    }

    fn decompress(&self, solution: &Vec<Decision>) -> Vec<Decision> {
        solution.clone()
    }
}