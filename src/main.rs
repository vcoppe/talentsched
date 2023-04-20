use clap::{Parser, Subcommand};
use resolution::Solve;

mod instance;
mod resolution;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct TalentSchedTools {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Solve(Solve)
}

fn main() {
    let cli = TalentSchedTools::parse();
    match cli.command {
        Command::Solve(solve) => solve.solve()
    }
}
