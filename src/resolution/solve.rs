use std::{fs::File, io::BufReader, time::Duration};

use clap::Args;
use ddo::{FixedWidth, TimeBudget, NoDupFringe, MaxUB, ParBarrierSolverFc, Completion, Solver};

use crate::resolution::model::{TalentSched, TalentSchedRelax, TalentSchedRanking};
use crate::instance::TalentSchedInstance;

#[derive(Debug, Args)]
pub struct Solve {
    /// The path to the instance file
    #[clap(short, long)]
    pub instance: String,
    /// max number of nodes in a layeer
    #[clap(short, long, default_value="100")]
    pub width: usize,
    /// timeout
    #[clap(short, long, default_value="60")]
    pub timeout: u64,
    /// If present, the path where to write the output html
    #[clap(short, long)]
    pub output: Option<String>,
}

impl Solve {
    pub fn solve(&self) {
        let instance: TalentSchedInstance = serde_json::from_reader(BufReader::new(File::open(&self.instance).unwrap())).unwrap();
        
        let problem = TalentSched::new(instance);
        let relaxation = TalentSchedRelax::new(problem.clone());

        let width = FixedWidth(self.width);
        let cutoff = TimeBudget::new(Duration::from_secs(self.timeout));
        let ranking = TalentSchedRanking;
        let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

        let mut solver = ParBarrierSolverFc::new(&problem, &relaxation, &ranking, &width, &cutoff, &mut fringe);

        let Completion{best_value, is_exact} = solver.maximize();

        let best_value = best_value.map(|v| -v).unwrap_or(isize::MAX);
        println!("is exact {is_exact}");
        println!("best value {best_value}");

        let mut sol = String::new();
        solver.best_solution().unwrap()
            .iter().map(|d| d.value)
            .for_each(|v| sol.push_str(&format!("{v} ")));

        println!("solution: {sol}");
    }
}