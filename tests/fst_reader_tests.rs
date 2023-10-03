// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use fst_native::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn hierarchy_to_str(entry: &FstHierarchyEntry) -> String {
    match entry {
        FstHierarchyEntry::Scope { name, .. } => format!("Scope: {name}"),
        FstHierarchyEntry::UpScope => format!("UpScope"),
        FstHierarchyEntry::Var { name, handle, .. } => format!("({handle}): {name}"),
        FstHierarchyEntry::AttributeBegin { name } => format!("BeginAttr: {name}"),
        FstHierarchyEntry::AttributeEnd => format!("EndAttr"),
    }
}

fn change_to_str(handle: FstSignalHandle, time: u64, value: &str) -> String {
    format!("{handle}@{time} = {value}")
}

fn run_dry_run(filename: &str, filter: &FstFilter) {
    let f = File::open(filename).expect(&format!("Failed to open {}", filename));
    let mut reader = FstReader::open(BufReader::new(f))
        .expect(&format!("Failed to read header from {}", filename));
    reader
        .read_hierarchy(|entry| println!("{}", hierarchy_to_str(&entry)))
        .expect(&format!("Failed to read hierarchy from {}", filename));
    reader
        .read_signals(filter, |time, handle, value| {
            println!("{}", &change_to_str(handle, time, value));
        })
        .expect(&format!("Failed to read data from {}", filename));
}

fn run_test(filename: &str, filter: &FstFilter) {
    let path = PathBuf::from(filename);
    let mut expected = path.clone();
    let f = File::open(path).expect(&format!("Failed to open {}", filename));

    expected.set_extension("expected.txt");
    let expected_f = File::open(expected).expect(&format!("Failed to open expected"));
    let mut expected_lines = BufReader::new(expected_f).lines();

    let mut reader = FstReader::open(BufReader::new(f))
        .expect(&format!("Failed to read header from {}", filename));
    let mut compare_line = |actual: &str| {
        if let Some(expected) = expected_lines.next() {
            assert_eq!(expected.unwrap(), actual);
        } else {
            panic!("Expected no more lines, but received:\n{actual}");
        }
    };
    reader
        .read_hierarchy(|entry| compare_line(&hierarchy_to_str(&entry)))
        .expect(&format!("Failed to read hierarchy from {}", filename));
    reader
        .read_signals(filter, |time, handle, value| {
            compare_line(&change_to_str(handle, time, value));
        })
        .expect(&format!("Failed to read data from {}", filename));
}

#[test]
fn read_verilator_basic_tests_anon() {
    run_test("fsts/VerilatorBasicTests_Anon.fst", &FstFilter::all());
}

#[test]
fn read_verilator_basic_tests_anon_time_filter() {
    // a higher start time should not affect anything (since there is only a single VC block in this file)
    let filter = FstFilter::filter_time(3, 7);
    run_test("fsts/VerilatorBasicTests_Anon.fst", &filter);
}
