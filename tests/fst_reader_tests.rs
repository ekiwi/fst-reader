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

    let f = std::fs::File::open("fsts/verilator/verilator-incomplete.fst")
        .unwrap_or_else(|_| panic!("Failed to open file"));
    let h = std::fs::File::open("fsts/verilator/verilator-incomplete.fst.hier")
        .unwrap_or_else(|_| panic!("Failed to open file"));
    let mut reader =
        FstReader::open_incomplete(std::io::BufReader::new(f), std::io::BufReader::new(h)).unwrap();

    load_header(&mut reader);
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
        let mut f = std::fs::File::open(filename.clone())
            .unwrap_or_else(|_| panic!("Failed to open {:?}", filename));
        let is_fst = is_fst_file(&mut f);
        let should_be_fst = true;
        assert_eq!(
            is_fst, should_be_fst,
            "{filename:?} should be detected as a FST! ({should_be_fst})"
        );
    }
}

#[test]
fn load_long_hierarchy_name() {
    // This file contains hierarchy names longer than 512 bytes, which is
    // the limit of the FST format (based on the specification).
    // However, e.g., Verilator can output longer names than that and this file used to throw an error.
    // See https://github.com/ekiwi/wellen/issues/49
    let f =
        std::fs::File::open("fsts/long_name.fst").unwrap_or_else(|_| panic!("Failed to open file"));

    let mut result = FstReader::open(std::io::BufReader::new(f)).unwrap();

    load_header(&mut result);
}

/// Test incomplete FST with 2 data sections reports correct end_time.
#[test]
fn test_multi_section_end_time_2sections() {
    let fst = std::fs::File::open("fsts/partial/minimal_2sections.fst").unwrap();
    let hier = std::fs::File::open("fsts/partial/minimal_2sections.fst.hier").unwrap();
    let mut reader =
        FstReader::open_incomplete(std::io::BufReader::new(fst), std::io::BufReader::new(hier))
            .unwrap();

    assert_eq!(reader.get_header().end_time, 200);

    let mut value_count = 0;
    reader
        .read_signals(&FstFilter::all(), |_, _, _| value_count += 1)
        .unwrap();
    assert_eq!(value_count, 63); // golden value from libfst
}

/// Test incomplete FST with 3 data sections reports correct end_time.
#[test]
fn test_multi_section_end_time_3sections() {
    let fst = std::fs::File::open("fsts/partial/minimal_3sections.fst").unwrap();
    let hier = std::fs::File::open("fsts/partial/minimal_3sections.fst.hier").unwrap();
    let mut reader =
        FstReader::open_incomplete(std::io::BufReader::new(fst), std::io::BufReader::new(hier))
            .unwrap();

    assert_eq!(reader.get_header().end_time, 300);

    let mut value_count = 0;
    reader
        .read_signals(&FstFilter::all(), |_, _, _| value_count += 1)
        .unwrap();
    assert_eq!(value_count, 93); // golden value from libfst
}

/// Test truncated 3-section FST reads 2 complete blocks before hitting truncated 3rd block.
#[test]
fn test_truncated_3sections_reads_2_blocks() {
    let fst = std::fs::File::open("fsts/partial/truncated_3sections.fst").unwrap();
    let hier = std::fs::File::open("fsts/partial/truncated_3sections.fst.hier").unwrap();
    let mut reader =
        FstReader::open_incomplete(std::io::BufReader::new(fst), std::io::BufReader::new(hier))
            .unwrap();

    assert_eq!(reader.get_header().end_time, 300);

    let mut value_count = 0;
    let result = reader.read_signals(&FstFilter::all(), |_, _, _| value_count += 1);

    assert!(
        matches!(&result, Err(ReaderError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof),
        "Expected UnexpectedEof from truncated block, got: {:?}",
        result
    );
    assert_eq!(value_count, 63);
}

/// Test that Real signals are correctly identified when using open_incomplete().
/// Regression test for bug where reconstruct_geometry() used length instead of type.
#[test]
fn test_incomplete_real_signal_type() {
    let fst = std::fs::File::open("fsts/partial/real_signal.fst").unwrap();
    let hier = std::fs::File::open("fsts/partial/real_signal.fst.hier").unwrap();
    let mut reader =
        FstReader::open_incomplete(std::io::BufReader::new(fst), std::io::BufReader::new(hier))
            .unwrap();

    reader.read_hierarchy(|_| {}).unwrap();

    let mut real_count = 0;
    let mut string_count = 0;
    reader
        .read_signals(&FstFilter::all(), |_, _, value| match value {
            FstSignalValue::Real(_) => real_count += 1,
            FstSignalValue::String(_) => string_count += 1,
        })
        .unwrap();

    assert_eq!(real_count, 10); // 10 Real signal changes
    assert_eq!(string_count, 10); // 10 BitVec signal changes (not Real)
}
