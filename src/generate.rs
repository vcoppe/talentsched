use std::{time::{SystemTime, UNIX_EPOCH}, fs::File, io::Write};

use clap::Args;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rand_distr::{Uniform, Distribution};

use crate::instance::TalentSchedInstance;

#[derive(Debug, Args)]
pub struct TalentSchedGenerator {
    /// An optional seed to kickstart the instance generation
    #[clap(short='s', long)]
    seed: Option<u128>,
    #[clap(short='n', long, default_value="20")]
    nb_scenes: usize,
    #[clap(short='p', long, default_value="5")]
    nb_actors: usize,
    /// The number of clusters of similar scene types
    #[clap(short='c', long, default_value="10")]
    nb_clusters: usize,
    #[clap(long, default_value="1")]
    min_cost: usize,
    #[clap(long, default_value="100")]
    max_cost: usize,
    #[clap(long, default_value="1")]
    min_duration: usize,
    #[clap(long, default_value="10")]
    max_duration: usize,
    /// The probability of needing an actor for a given scene
    #[clap(short='d', long, default_value="0.5")]
    density: f64,
    #[clap(long, default_value="0.7")]
    similarity: f64,
    /// Name of the file where to generate the talentsched instance
    #[clap(short, long)]
    output: Option<String>,
}

impl TalentSchedGenerator {

    pub fn generate(&mut self) {
        let mut rng = self.rng();

        let mut nb_scenes_per_cluster = vec![self.nb_scenes / self.nb_clusters; self.nb_clusters];
        for i in 0..(self.nb_scenes % self.nb_clusters) {
            nb_scenes_per_cluster[i] += 1;
        }
        
        let cost = self.generate_costs(&mut rng);
        let duration = self.generate_duration(&mut rng);
        let actors = self.generate_actors(&mut rng, &nb_scenes_per_cluster);

        let instance = TalentSchedInstance {
            nb_scenes: self.nb_clusters,
            nb_actors: self.nb_actors,
            cost,
            duration,
            actors,
        };

        let instance = serde_json::to_string_pretty(&instance).unwrap();

        if let Some(output) = self.output.as_ref() {
            File::create(output).unwrap().write_all(instance.as_bytes()).unwrap();
        } else {
            println!("{instance}");
        }
    }

    fn generate_costs(&self, rng: &mut impl Rng) -> Vec<usize> {
        let mut costs = vec![];

        let rand_cost = Uniform::new_inclusive(self.min_cost, self.max_cost);
        for _ in 0..self.nb_clusters {
            costs.push(rand_cost.sample(rng));
        }

        costs
    }

    fn generate_duration(&self, rng: &mut impl Rng) -> Vec<usize> {
        let mut durations = vec![];

        let rand_duration = Uniform::new_inclusive(self.min_duration, self.max_duration);
        for _ in 0..self.nb_clusters {
            durations.push(rand_duration.sample(rng));
        }

        durations
    }

    fn generate_actors(&self, rng: &mut impl Rng, nb_scenes_per_cluster: &Vec<usize>) -> Vec<Vec<usize>> {
        let mut actors = vec![vec![0; self.nb_scenes]; self.nb_actors];

        let rand_duration = Uniform::new_inclusive(0.0, 1.0);

        let mut scene = 0;
        for nb_scenes in nb_scenes_per_cluster {
            for i in 0..self.nb_actors {
                actors[i][scene] = if rand_duration.sample(rng) < self.density {
                    1
                } else {
                    0
                };

                for j in 1..*nb_scenes {
                    actors[i][scene+j] = if rand_duration.sample(rng) < self.similarity {
                        1
                    } else {
                        0
                    };
                }
            }
            scene += *nb_scenes;
        }

        actors
    }

    fn rng(&self) -> impl Rng {
        let init = self.seed.unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis());
        let mut seed = [0_u8; 32];
        seed.iter_mut().zip(init.to_be_bytes().into_iter()).for_each(|(s, i)| *s = i);
        seed.iter_mut().rev().zip(init.to_le_bytes().into_iter()).for_each(|(s, i)| *s = i);
        ChaChaRng::from_seed(seed)
    }

}
