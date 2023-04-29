use std::collections::HashMap;

use ddo::{Compression, Problem, Decision};
use smallbitset::Set64;

use crate::instance::TalentSchedInstance;

use super::model::{TalentSched, TalentSchedState};

pub struct TalentSchedCompression<'a> {
    pub problem: &'a TalentSched,
    pub meta_problem: TalentSched,
    pub membership: HashMap<isize, isize>,
    pub members: Vec<Set64>,
    pub size: Vec<usize>,
}

impl<'a> TalentSchedCompression<'a> {
    pub fn new(problem: &'a TalentSched, n_meta_scenes: usize) -> Self {
        let membership = Self::cluster_scenes(problem, n_meta_scenes);

        let duration = Self::compute_meta_duration(problem, &membership, n_meta_scenes);
        let actors = Self::compute_meta_actors(problem, &membership, n_meta_scenes);
        
        let meta_instance = TalentSchedInstance {
            nb_scenes: n_meta_scenes,
            nb_actors: problem.instance.nb_actors,
            cost: problem.instance.cost.clone(),
            duration,
            actors,
        };

        let meta_problem = TalentSched::new(meta_instance);

        let mut mapping = HashMap::new();
        let mut members = vec![Set64::default(); n_meta_scenes];
        let mut size = vec![0; n_meta_scenes];
        for (i, j) in membership.iter().copied().enumerate() {
            mapping.insert(i as isize, j as isize);
            members[j].add_inplace(i);
            size[j] += 1;
        }

        TalentSchedCompression {
            problem,
            meta_problem,
            membership: mapping,
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
                meta_actors[actor][j] &= pb.instance.actors[actor][i];
            }
        }

        meta_actors
    }

    fn cluster_scenes(pb: &TalentSched, n_meta_scenes: usize) -> Vec<usize> {
        let mut clusters = vec![];
        (0..pb.instance.nb_scenes).for_each(|i| {
            let mut cluster = Set64::default();
            cluster.add_inplace(i);
            clusters.push((pb.actors[i], cluster));
        });

        while clusters.len() > n_meta_scenes {
            let mut min_loss = (usize::MAX, 0, 0);

            for (i, a) in clusters.iter().enumerate() {
                for (j, b) in clusters.iter().enumerate().skip(i+1) {

                    let actors = a.0.inter(b.0);

                    let mut loss = 0;
                    for s in a.1.iter() {
                        for k in pb.actors[s].iter() {
                            if !actors.contains(k) {
                                loss += pb.instance.cost[k] * pb.instance.duration[s];
                            }
                        }
                    }
                    for s in b.1.iter() {
                        for k in pb.actors[s].iter() {
                            if !actors.contains(k) {
                                loss += pb.instance.cost[k] * pb.instance.duration[s];
                            }
                        }
                    }

                    if loss < min_loss.0 {
                        min_loss = (loss, i, j);
                    }
                }
            }

            let cluster = clusters.remove(min_loss.2);

            clusters[min_loss.1].0.inter_inplace(&cluster.0);
            clusters[min_loss.1].1.union_inplace(&cluster.1);
        }

        let mut membership = vec![0; pb.instance.nb_scenes];
        for (i, cluster) in clusters.iter().enumerate() {
            for j in cluster.1.iter() {
                membership[j] = i;
            }
        }
 
        membership
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

        if !state.maybe_scenes.is_empty() {
            return compressed;
        }

        for i in 0..self.meta_problem.instance.nb_scenes {
            let present_from_cluster = state.scenes.inter(self.members[i]);
            if present_from_cluster == self.members[i] {
                compressed.scenes.add_inplace(i);
            } else if !present_from_cluster.is_empty() {
                compressed.scenes = Set64::default();
                break;
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