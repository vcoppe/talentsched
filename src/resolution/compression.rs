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
    pub cluster_membership: Vec<usize>,
    pub membership: HashMap<isize, isize>,
    pub members: Vec<Set64>,
    pub n_last_members: Vec<Vec<Set64>>,
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

        let mut n_last_members = members.iter().map(|m| vec![Set64::empty(); m.len() + 1]).collect::<Vec<Vec<Set64>>>();
        for (cluster, m) in members.iter().enumerate() {
            for (i, j) in m.iter().enumerate() {
                for k in 0..(i+1) {
                    n_last_members[cluster][m.len() - k].add_inplace(j);
                }
            }
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
            cluster_membership: clustering.membership,
            membership,
            members,
            n_last_members,
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
        let mut compressed = TalentSchedState {
            scenes: Set64::default(),
            maybe_scenes: Set64::default(),
        };

        if state.maybe_scenes.is_empty() {
            let mut n_from_cluster = vec![0; self.members.len()];
            for i in state.scenes.iter() {
                n_from_cluster[self.cluster_membership[i]] += 1;
            }

            for (i, n) in n_from_cluster.iter().copied().enumerate() {
                compressed.scenes.union_inplace(&self.n_last_members[i][n]);
            }
        }

        compressed
    }

    fn decompress(&self, solution: &Vec<Decision>) -> Vec<Decision> {
        solution.clone()
    }
}