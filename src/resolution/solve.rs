use std::fmt::Display;
use std::str::FromStr;
use std::sync::Arc;
use std::{fs::File, io::BufReader};
use std::hash::Hash;

use clap::Args;
use ddo::{CompressedSolutionBound, DecisionHeuristicBuilder, NoHeuristicBuilder, CompressedSolutionHeuristicBuilder, Problem, SubProblem, CompilationInput, NoCutoff, CompilationType, FRONTIER, NoHeuristic, Mdd, VizConfigBuilder, DecisionDiagram, EmptyBarrier};

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
        Box::new(TalentSchedRelax::new(compressor.problem.clone(), Some(CompressedSolutionBound::new(compressor))))
    } else {
        Box::new(TalentSchedRelax::new(compressor.problem.clone(), None))
    }
}

fn get_heuristic<'a>(compressor: &'a TalentSchedCompression, compression_heuristic: bool) -> Box<dyn DecisionHeuristicBuilder<TalentSchedState> + Send + Sync + 'a> {
    if compression_heuristic {
        Box::new(CompressedSolutionHeuristicBuilder::new(compressor, &compressor.membership))
    } else {
        Box::new(NoHeuristicBuilder::default())
    }
}

impl Solve {
    pub fn solve(&self) {
        let instance: TalentSchedInstance = serde_json::from_reader(BufReader::new(File::open(&self.instance).unwrap())).unwrap();
        
        let problem = TalentSched::new(instance);

        let compressor = TalentSchedCompression::new(&problem, self.n_meta_items);
        let relaxation = get_relaxation(&compressor, self.compression_bound);
        let heuristic = get_heuristic(&compressor, self.compression_heuristic);

        let ranking = TalentSchedRanking;

        let barrier = EmptyBarrier::default();

        let residual = SubProblem { 
            state: Arc::new(problem.initial_state()), 
            value: 0, 
            path: vec![], 
            ub: isize::MAX, 
            depth: 0
        };
        let input = CompilationInput {
            comp_type: CompilationType::Relaxed,
            problem: &problem,
            relaxation: relaxation.as_ref(),
            ranking: &ranking,
            cutoff: &NoCutoff,
            max_width: self.width,
            residual: &residual,
            best_lb: isize::MIN,
            barrier: &barrier,
            heuristic: heuristic.get_heuristic(&problem.initial_state()),
        };

        let mut clean = Mdd::<TalentSchedState, {FRONTIER}>::new();
        _ = clean.compile(&input);

        let config = VizConfigBuilder::default()
            .show_deleted(true)
            .group_merged(true)
            .build()
            .unwrap();
        
        let dot = clean.as_graphviz(&config);

        println!("original DD");
        println!("{dot}");
        
        print!("clustering: ");
        compressor.members.iter().for_each(|m| print!(" {m}"));
        println!();
        println!("clustered instance");
        println!("{}", serde_json::to_string_pretty(&compressor.meta_problem.instance).unwrap());

        let residual = SubProblem { 
            state: Arc::new(compressor.meta_problem.initial_state()), 
            value: 0, 
            path: vec![], 
            ub: isize::MAX, 
            depth: 0
        };
        let input = CompilationInput {
            comp_type: CompilationType::Relaxed,
            problem: &compressor.meta_problem,
            relaxation: relaxation.as_ref(),
            ranking: &ranking,
            cutoff: &NoCutoff,
            max_width: usize::MAX,
            residual: &residual,
            best_lb: isize::MIN,
            barrier: &barrier,
            heuristic: Arc::new(NoHeuristic),
        };

        let mut clean = Mdd::<TalentSchedState, {FRONTIER}>::new();
        _ = clean.compile(&input);

        let config = VizConfigBuilder::default()
            .show_deleted(true)
            .group_merged(true)
            .build()
            .unwrap();
        
        let dot = clean.as_graphviz(&config);

        println!("compressed DD");
        println!("{dot}");

    }
}