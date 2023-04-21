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
        // (self.pb.instance.actors[i][self.id] * self.pb.instance.cost[i]) as f64
    }
}

pub struct TalentSchedCompression<'a> {
    pub problem: &'a TalentSched,
    pub meta_problem: TalentSched,
    pub membership: HashMap<isize, isize>,
    pub members: Vec<Set64>,
    pub size: Vec<usize>,
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

        let duration = Self::compute_meta_duration(problem, &clustering.membership, n_meta_scenes);
        let actors = Self::compute_meta_actors(problem, &clustering.membership, n_meta_scenes);
        
        let meta_instance = TalentSchedInstance {
            nb_scenes: n_meta_scenes,
            nb_actors: problem.instance.nb_actors,
            cost: problem.instance.cost.clone(),
            duration,
            actors,
        };

        let meta_problem = TalentSched::new(meta_instance, Some(problem.forced_cost));

        let mut membership = HashMap::new();
        let mut members = vec![Set64::default(); n_meta_scenes];
        let mut size = vec![0; n_meta_scenes];
        for (i, j) in clustering.membership.iter().enumerate() {
            membership.insert(i as isize, *j as isize);
            members[*j].add_inplace(i);
            size[*j] += 1;
        }

        TalentSchedCompression {
            problem,
            meta_problem,
            membership,
            members,
            size,
        }
    }

    fn compute_meta_duration(pb: &TalentSched, membership: &Vec<usize>, n_meta_scenes: usize) -> Vec<usize> {
        let mut meta_duration = vec![0; n_meta_scenes];
        
        for (i, j) in membership.iter().copied().enumerate() {
            meta_duration[j] += pb.instance.duration[i];
        }

        meta_duration
    }

    fn compute_meta_actors(pb: &TalentSched, membership: &Vec<usize>, n_meta_scenes: usize) -> Vec<Vec<usize>> {
        let mut meta_actors = vec![vec![1; n_meta_scenes]; pb.instance.nb_actors];

        for (i, j) in membership.iter().copied().enumerate() {
            for actor in 0..pb.instance.nb_actors {
                meta_actors[actor][j] = meta_actors[actor][j].min(pb.instance.actors[actor][i]);
            }
        }

        meta_actors
    }
}

impl<'a> Compression for TalentSchedCompression<'a> {
    type State = TalentSchedState;

    fn get_compressed_problem(&self) -> &dyn Problem<State = Self::State> {
        &self.meta_problem
    }

    fn compress(&self, state: &TalentSchedState) -> TalentSchedState {
        let mut compressed = TalentSchedState {
            scenes: Set64::default(),
            maybe_scenes: Set64::default(),
        };
        
        for i in 0..self.meta_problem.instance.nb_scenes {
            if state.scenes.inter(self.members[i]) == self.members[i] {
                compressed.scenes.add_inplace(i);
            }
        }

        compressed
    }

    fn decompress(&self, solution: &Vec<Decision>) -> Vec<Decision> {
        let mut sol = vec![];

        for decision in solution {
            for _ in 0..self.size[decision.value as usize] {
                sol.push(Decision { variable: decision.variable, value: decision.value });
            }
        }

        let start = self.problem.instance.nb_scenes - sol.len();
        for (i, decision) in sol.iter_mut().enumerate() {
            decision.variable.0 = start + i;
        }

        sol
    }
}