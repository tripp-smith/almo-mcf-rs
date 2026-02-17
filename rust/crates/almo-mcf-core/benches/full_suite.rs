use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem};
use std::time::Instant;

fn tiny_problem() -> McfProblem {
    McfProblem::new(vec![0], vec![1], vec![0], vec![10], vec![1], vec![-5, 5]).expect("valid")
}

fn parse_runs() -> usize {
    let mut runs = 1usize;
    let args: Vec<String> = std::env::args().collect();
    let mut idx = 0;
    while idx < args.len() {
        if args[idx] == "--runs" && idx + 1 < args.len() {
            if let Ok(parsed) = args[idx + 1].parse::<usize>() {
                runs = parsed.max(1);
            }
            idx += 1;
        }
        idx += 1;
    }
    runs
}

fn main() {
    let runs = parse_runs();
    let problem = tiny_problem();
    let opts = McfOptions {
        use_ipm: Some(false),
        ..McfOptions::default()
    };

    let mut total = 0.0;
    for _ in 0..runs {
        let start = Instant::now();
        let _ = min_cost_flow_exact(&problem, &opts).expect("solve should succeed");
        total += start.elapsed().as_secs_f64();
    }

    let exact_avg = total / runs as f64;
    println!("{{\"runs\":{runs},\"exact_avg_s\":{exact_avg:.6}}}");
}
