use clap::{Parser, Subcommand};
use generate::TalentSchedGenerator;
use resolution::Solve;

mod instance;
mod generate;
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
    Generate(TalentSchedGenerator),
    Solve(Solve)
}

fn main() {
    let cli = TalentSchedTools::parse();
    match cli.command {
        Command::Generate(mut generate) => generate.generate(),
        Command::Solve(solve) => solve.solve()
    }
}
