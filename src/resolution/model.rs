use std::vec;

use ddo::*;
use ordered_float::OrderedFloat;
use smallbitset::Set64;

use crate::instance::TalentSchedInstance;

/// The state of the DP model
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TalentSchedState {
    pub scenes: Set64,
    pub maybe_scenes: Set64,
}

/// This structure describes a TalentSched instance
#[derive(Debug, Clone)]
pub struct TalentSched {
    pub instance: TalentSchedInstance,
    pub actors: Vec<Set64>,
}

impl TalentSched {
    pub fn new(instance: TalentSchedInstance) -> Self {
        let mut actors = vec![Set64::default(); instance.nb_scenes];

        for i in 0..instance.nb_actors {
            for j in 0..instance.nb_scenes {
                if instance.actors[i][j] == 1 {
                    actors[j].add_inplace(i);
                }
            }
        }

        TalentSched {instance, actors }
    }

    fn get_present(&self, state: &TalentSchedState) -> Set64 {
        let mut before = Set64::default();
        let mut after = Set64::default();

        for i in 0..self.instance.nb_scenes {
            if !state.maybe_scenes.contains(i) {
                if state.scenes.contains(i) {
                    after.union_inplace(&self.actors[i]);
                } else {
                    before.union_inplace(&self.actors[i]);
                }
            }
        }

        before.inter(after)
    }
}

impl Problem for TalentSched {
    type State = TalentSchedState;

    fn nb_variables(&self) -> usize {
        self.instance.nb_scenes
    }

    fn initial_state(&self) -> Self::State {
        let mut scenes = Set64::default();
        for i in 0..self.instance.nb_scenes {
            scenes.add_inplace(i);
        }

        TalentSchedState {
            scenes,
            maybe_scenes: Default::default(),
        }
    }

    fn initial_value(&self) -> isize {
        0
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        let mut ret = state.clone();
        
        ret.scenes.remove_inplace(decision.value as usize);
        ret.maybe_scenes.remove_inplace(decision.value as usize);

        ret
    }

    fn transition_cost(&self, state: &Self::State, decision: ddo::Decision) -> isize {
        let scene = decision.value as usize;

        let pay = self.get_present(state).union(self.actors[scene]);

        let mut cost = 0;
        for actor in pay.iter() {
            cost += self.instance.cost[actor] * self.instance.duration[scene];
        }

        - (cost as isize)
    }

    fn next_variable(&self, depth: usize, _: &mut dyn Iterator<Item = &Self::State>)
        -> Option<ddo::Variable> {
        if depth < self.instance.nb_scenes {
            Some(Variable(depth))
        } else {
            None
        }
    }

    fn for_each_in_domain(&self, variable: ddo::Variable, state: &Self::State, f: &mut dyn ddo::DecisionCallback) {
        let mut count = 0;

        for i in state.scenes.iter() {
            f.apply(Decision { variable, value: i as isize });
            count += 1;
        }

        if variable.id() + count < self.instance.nb_scenes {
            for i in state.maybe_scenes.iter() {
                f.apply(Decision { variable, value: i as isize });
            }
        }
    }
}

/// This structure implements the TalentSched relaxation
pub struct TalentSchedRelax<'a> {
    pb: TalentSched,
    pub compression_bound: Option<CompressedSolutionBound<'a, TalentSchedState>>,
}

impl<'a> TalentSchedRelax<'a> {
    pub fn new(pb: TalentSched, compression_bound: Option<CompressedSolutionBound<'a, TalentSchedState>>) -> Self {
        Self { pb, compression_bound }
    }
}

impl<'a> Relaxation for TalentSchedRelax<'a> {
    type State = TalentSchedState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut merged = states.next().unwrap().clone();
        merged.maybe_scenes.union_inplace(&merged.scenes);

        for s in states {
            merged.scenes.inter_inplace(&s.scenes);
            merged.maybe_scenes.union_inplace(&s.scenes);
            merged.maybe_scenes.union_inplace(&s.maybe_scenes);
        }

        merged.maybe_scenes.diff_inplace(&merged.scenes);
        
        merged
    }

    fn relax(
        &self,
        _source: &Self::State,
        _dest: &Self::State,
        _new:  &Self::State,
        _decision: Decision,
        cost: isize,
    ) -> isize {
        cost
    }

    fn fast_upper_bound(&self, state: &Self::State) -> isize {
        let mut lb = 0.0;

        let present_actors = self.pb.get_present(state);
        let mut r = (0..self.pb.instance.nb_actors).map(|i| (OrderedFloat(0.0), i)).collect::<Vec<(OrderedFloat<f64>, usize)>>();

        for scene in state.scenes.iter() {
            let present_actors_from_scene = self.pb.actors[scene].inter(present_actors);
            if !present_actors_from_scene.is_empty() {
                let mut total_cost = 0.0;
                let mut total_cost_sq = 0.0;

                for actor in present_actors_from_scene.iter() {
                    total_cost += self.pb.instance.cost[actor] as f64;
                    total_cost_sq += (self.pb.instance.cost[actor] * self.pb.instance.cost[actor]) as f64;
                }

                for actor in present_actors_from_scene.iter() {
                    r[actor].0 += self.pb.instance.duration[scene] as f64 / total_cost;
                }
                lb -= self.pb.instance.duration[scene] as f64 * (total_cost + total_cost_sq / total_cost) / 2.0;
            }

            for actor in self.pb.actors[scene].iter() {
                lb += (self.pb.instance.cost[actor] * self.pb.instance.duration[scene]) as f64;
            }
        }

        r.sort_unstable();

        let mut sum_e = 0.0;
        for (r_a, a) in r {
            if present_actors.contains(a) {
                sum_e += r_a.0 * self.pb.instance.cost[a] as f64;
                lb += self.pb.instance.cost[a] as f64 * sum_e;
            }
        }

        if lb - lb.floor() < 1e-2 {
            lb = lb.floor();
        } else {
            lb = lb.ceil();
        }

        let mut rub = - (lb as isize);
        if let Some(bound) = &self.compression_bound {
            rub = rub.min(bound.get_ub(state));
        }
        rub
    }
}


/// The last bit of information which we need to provide when implementing a ddo-based
/// solver is a `StateRanking`. This is an heuristic which is used to select the most
/// and least promising nodes as a means to only delete/merge the *least* promising nodes
/// when compiling restricted and relaxed DDs.
pub struct TalentSchedRanking;
impl StateRanking for TalentSchedRanking {
    type State = TalentSchedState;

    fn compare(&self, a: &Self::State, b: &Self::State) -> std::cmp::Ordering {
        let tot_a = a.scenes.len() + a.maybe_scenes.len();
        let tot_b = b.scenes.len() + b.maybe_scenes.len();
        
        tot_a.cmp(&tot_b)
    }
}
