// Copyright 2023 The Regents of the University of California
// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// It is easiest to write a Diff test. However, on some inputs GTKWave actually crashes
// and thus we cannot compare.

use fst_reader::*;
use std::io::{BufRead, Seek};
use std::path::{Path, PathBuf};

mod utils;
use utils::hierarchy_to_str;

fn run_load_test(filename: &str, _filter: &FstFilter) {
    let f = std::fs::File::open(filename).unwrap_or_else(|_| panic!("Failed to open {}", filename));
    let mut reader = FstReader::open(std::io::BufReader::new(f)).unwrap();

    load_header(&mut reader);
}

fn load_header<R: BufRead + Seek>(reader: &mut FstReader<R>) -> Vec<String> {
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

#[test]
fn load_verilator_incomplete() {
    let f = std::fs::File::open("fsts/verilator/verilator-incomplete.fst")
        .unwrap_or_else(|_| panic!("Failed to open file"));

    let result = FstReader::open(std::io::BufReader::new(f));
    assert!(matches!(result, Err(ReaderError::MissingGeometry())));
}

#[test]
fn load_time_table_treadle_gcd() {
    let filename = "fsts/treadle/GCD.vcd.fst";
    let f = std::fs::File::open(filename).unwrap_or_else(|_| panic!("Failed to open {}", filename));
    let reader = FstReader::open_and_read_time_table(std::io::BufReader::new(f)).unwrap();
    let expected = [0u64, 1, 2, 3, 4];
    assert_eq!(reader.get_time_table().unwrap(), expected);
}

fn find_fst_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().filter_map(Result::ok) {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            let mut sub = find_fst_files(&entry_path);
            out.append(&mut sub);
        }
        if entry_path.to_str().unwrap().ends_with(".fst") {
            out.push(entry_path);
        }
    }
    out.sort();
    out
}

#[test]
fn test_is_fst_file() {
    let fsts = find_fst_files(Path::new("fsts/"));
    for filename in fsts {
        dbg!(&filename);
        let mut f = std::fs::File::open(filename.clone())
            .unwrap_or_else(|_| panic!("Failed to open {:?}", filename));
        let is_fst = is_fst_file(&mut f);
        dbg!(is_fst);
        let should_be_fst = true;
        assert_eq!(
            is_fst, should_be_fst,
            "{filename:?} should be detected as a FST! ({should_be_fst})"
        );
    }
}
