use std::path::PathBuf;

use denning::MissRatioCurve;
use palc::Parser;

#[derive(serde::Serialize, serde::Deserialize)]
struct Data {
    pub miss_ratio_curve: MissRatioCurve,
    /// Access total of the analysed program, carried through unchanged so
    /// `mrc` can turn the converted ratio into a miss count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_count: Option<String>,
}

#[derive(palc::Parser)]
struct Cli {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,
    /// Output file path (optional, defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Associativity to convert to
    #[arg(short, long)]
    assoc: usize,
}

fn main() {
    let cli = Cli::parse();
    let input: Data =
        simd_json::from_reader(std::fs::File::open(&cli.input).expect("Failed to open input file"))
            .expect("Failed to parse input JSON");
    let data = Data {
        miss_ratio_curve: input
            .miss_ratio_curve
            .compute_assoc(cli.assoc),
        total_count: input.total_count,
    };
    let output: Box<dyn std::io::Write> = if let Some(output_path) = cli.output {
        let file = std::fs::File::create(output_path).expect("Failed to create output file");
        Box::new(file)
    } else {
        Box::new(std::io::stdout())
    };
    simd_json::to_writer_pretty(output, &data).expect("Failed to write output JSON");
}
