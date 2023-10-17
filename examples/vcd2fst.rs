// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "vcd2fst")]
#[command(author = "Kevin Laeufer <laeufer@berkeley.edu>")]
#[command(version)]
#[command(about = "Converts a VCD file to an FST file.", long_about = None)]
struct Args {
    #[arg(value_name = "VCDFILE", index = 1)]
    vcd_file: String,
    #[arg(value_name = "FSTFILE", index = 2)]
    fst_file: String,
}

fn main() {
    let args = Args::parse();

    let input = std::fs::File::open(args.vcd_file).expect("failed to open input file!");
    let mut reader = vcd::Parser::new(std::io::BufReader::new(input));
    let _header = reader.parse_header().expect("failed to parse header");
}
