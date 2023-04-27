use std::fmt::Display;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use std::{fs::File, io::BufReader, time::Duration};
use std::hash::Hash;

use clap::Args;
use ddo::{FixedWidth, TimeBudget, NoDupFringe, MaxUB, ParBarrierSolverFc, Completion, Solver, CompressedSolutionBound, DecisionHeuristicBuilder, NoHeuristicBuilder, CompressedSolutionHeuristicBuilder, SimpleBarrier, HybridSolver, WidthHeuristic, Problem, Relaxation, StateRanking, Cutoff, Fringe, FullMdd};

use crate::resolution::model::{TalentSched, TalentSchedRelax, TalentSchedRanking};
use crate::instance::TalentSchedInstance;

use super::compression::TalentSchedCompression;
use super::model::TalentSchedState;

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
    /// number of threads used
    #[clap(long, default_value="1")]
    pub threads: usize,
    /// The number of item clusters
    #[clap(short, long, default_value="10")]
    pub n_meta_items: usize,
    /// Whether to use the compression-based bound
    #[clap(short='b', long, action)]
    pub compression_bound: bool,
    /// Whether to use the compression-based decision heuristic
    #[clap(short='h', long, action)]
    pub compression_heuristic: bool,
    /// The solver to use
    #[clap(short, long, default_value="classic")]
    pub solver: SolverType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverType {
    Classic,
    Hybrid,
}
impl FromStr for SolverType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "classic" => Ok(Self::Classic),
            "hybrid"  => Ok(Self::Hybrid),
            _ => Err("The only supported frontier types are 'classic' and 'hybrid'"),
        }
    }
}
impl Display for SolverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Classic => write!(f, "classic"),
            Self::Hybrid  => write!(f, "hybrid"),
        }
    }
}

fn get_relaxation<'a>(compressor: &'a TalentSchedCompression, compression_bound: bool) -> Box<TalentSchedRelax<'a>> {
    if compression_bound {
        Box::new(TalentSchedRelax::new(compressor.problem.clone(), Some(CompressedSolutionBound::new(compressor, None))))
    } else {
        Box::new(TalentSchedRelax::new(compressor.problem.clone(), None))
    }
}

fn get_heuristic<'a>(compressor: &'a TalentSchedCompression, compression_heuristic: bool, solutions: Option<Arc<FullMdd<TalentSchedState>>>) -> Box<dyn DecisionHeuristicBuilder<TalentSchedState> + Send + Sync + 'a> {
    if compression_heuristic {
        Box::new(CompressedSolutionHeuristicBuilder::new(compressor, &compressor.membership, solutions))
    } else {
        Box::new(NoHeuristicBuilder::default())
    }
}

fn get_solver<'a, State>(
    solver: SolverType,
    threads: usize,
    problem: &'a (dyn Problem<State = State> + Send + Sync),
    relaxation: &'a (dyn Relaxation<State = State> + Send + Sync),
    ranking: &'a (dyn StateRanking<State = State> + Send + Sync),
    width: &'a (dyn WidthHeuristic<State> + Send + Sync),
    cutoff: &'a (dyn Cutoff + Send + Sync), 
    fringe: &'a mut (dyn Fringe<State = State> + Send + Sync),
    heuristic_builder: &'a (dyn DecisionHeuristicBuilder<State> + Send + Sync),

) -> Box<dyn Solver + 'a>
where State: Eq + Hash + Clone + Send + Sync
{
    match solver {
        SolverType::Classic => {
            Box::new(ParBarrierSolverFc::custom(
                problem,
                relaxation,
                ranking,
                width,
                cutoff,
                fringe,
                threads,
                heuristic_builder
            ))
        },
        SolverType::Hybrid => {
            Box::new(HybridSolver::<State, SimpleBarrier<State>>::custom(
                problem,
                relaxation,
                ranking,
                width,
                cutoff,
                fringe,
                threads,
                heuristic_builder
            ))
        },
    }
}

impl Solve {
    pub fn solve(&self) {
        let instance: TalentSchedInstance = serde_json::from_reader(BufReader::new(File::open(&self.instance).unwrap())).unwrap();
        
        let problem = TalentSched::new(instance);

        let compressor = TalentSchedCompression::new(&problem, self.n_meta_items);
        let relaxation = get_relaxation(&compressor, self.compression_bound);
        let solutions = match &relaxation.compression_bound {
            Some(bd) => Some(bd.compressed_solutions.clone()),
            None => None,
        };
        let heuristic = get_heuristic(&compressor, self.compression_heuristic, solutions);

        let width = FixedWidth(self.width);
        let cutoff = TimeBudget::new(Duration::from_secs(self.timeout));
        let ranking = TalentSchedRanking;
        let mut fringe = NoDupFringe::new(MaxUB::new(&ranking));

        let mut solver = get_solver(
            self.solver,
            self.threads,
            &problem,
            relaxation.as_ref(),
            &ranking,
            &width,
            &cutoff,
            &mut fringe,
            heuristic.as_ref()
        );

        let start = Instant::now();

        let Completion{best_value, is_exact} = solver.maximize();

        let duration = start.elapsed();

        let best_value = best_value.map(|v| -v).unwrap_or(isize::MAX);
        let best_bound = -solver.best_upper_bound();
        let explored = solver.explored();

        let mut sol = String::new();
        solver.best_solution().unwrap()
            .iter().map(|d| d.value)
            .for_each(|v| sol.push_str(&format!("{v} ")));

        println!("===== settings =====");
        println!("solver     : {}", self.solver);
        println!("cmpr. bound: {}", self.compression_bound);
        println!("cmpr. heu. : {}", self.compression_heuristic);
        println!("===== results  =====");
        println!("is exact   : {is_exact}");
        println!("best value : {best_value}");
        println!("best bound : {best_bound}");
        println!("duration   : {:.3} seconds", duration.as_secs_f32());
        println!("explored   : {explored}");
        println!("solution   : {sol}");
    }
}