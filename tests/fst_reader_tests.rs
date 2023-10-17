// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// It is easiest to write a Diff test. However, on some inputs GTKWave actually crashes
// and thus we cannot compare.

use fst_native::*;
use std::io::{Read, Seek};

mod utils;
use utils::hierarchy_to_str;

fn run_load_test(filename: &str, filter: &FstFilter) {
    let f = std::fs::File::open(filename).unwrap_or_else(|_| panic!("Failed to open {}", filename));
    let mut reader = FstReader::open(f).unwrap();

    load_header(&mut reader);
}

fn load_header<R: Read + Seek>(reader: &mut FstReader<R>) -> Vec<String> {
    let mut is_real = Vec::new();
    let mut hierarchy = Vec::new();
    let foo = |entry: FstHierarchyEntry| {
        // remember if variables are real valued
        match &entry {
            FstHierarchyEntry::Var { tpe, handle, .. } => {
                let is_var_real = match tpe {
                    FstVarType::Real
                    | FstVarType::RealParameter
                    | FstVarType::RealTime
                    | FstVarType::ShortReal => true,
                    _ => false,
                };
                let idx = handle.get_index();
                if is_real.len() <= idx {
                    is_real.resize(idx + 1, false);
                }
                is_real[idx] = is_var_real;
            }
            _ => {}
        };

        let actual = hierarchy_to_str(&entry);
        hierarchy.push(actual);
    };
    reader.read_hierarchy(foo).unwrap();
    hierarchy
}

// GTKWave actually crashes on this input, even though it was created by vcd2fst from GTKWave
#[test]
#[ignore] // TODO: do not crash, fail more gracefully
fn load_sigrok() {
    run_load_test("fsts/sigrok/libsigrok.vcd.fst", &FstFilter::all());
}
