use std::vec;

use ddo::*;
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
        let mut cost = 0;
        for (scene, actors) in self.actors.iter().enumerate() {
            for actor in actors.iter() {
                cost += self.instance.cost[actor] * self.instance.duration[scene];
            }
        }
        - (cost as isize)
    }

    fn transition(&self, state: &Self::State, decision: ddo::Decision) -> Self::State {
        let mut ret = state.clone();
        
        ret.scenes.remove_inplace(decision.value as usize);
        ret.maybe_scenes.remove_inplace(decision.value as usize);

        ret
    }

    fn transition_cost(&self, state: &Self::State, decision: ddo::Decision) -> isize {
        let scene = decision.value as usize;

        let pay = self.get_present(state).diff(self.actors[scene]);

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
                count += 1;
            }
        }
    }
}

/// This structure implements the TalentSched relaxation
pub struct TalentSchedRelax {
    pb: TalentSched,
}

impl TalentSchedRelax {
    pub fn new(pb: TalentSched) -> Self {
        Self { pb }
    }
}

impl Relaxation for TalentSchedRelax {
    type State = TalentSchedState;

    fn merge(&self, states: &mut dyn Iterator<Item = &Self::State>) -> Self::State {
        let mut merged = states.next().unwrap().clone();

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

    fn fast_upper_bound(&self, _state: &Self::State) -> isize {
        0
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
