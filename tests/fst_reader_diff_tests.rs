// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use fst_native::*;
use std::ffi::{c_void, CStr, CString};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn fst_sys_load_header(handle: *mut c_void) -> FstHeader {
    unsafe {
        let version = fst_sys::fstReaderGetVersionString(handle);
        let date = fst_sys::fstReaderGetDateString(handle);
        FstHeader {
            start_time: fst_sys::fstReaderGetStartTime(handle),
            end_time: fst_sys::fstReaderGetEndTime(handle),
            var_count: fst_sys::fstReaderGetVarCount(handle),
            max_handle: fst_sys::fstReaderGetMaxHandle(handle) as u64,
            version: CStr::from_ptr(version).to_str().unwrap().to_string(),
            date: CStr::from_ptr(date).to_str().unwrap().to_string(),
        }
    }
}

fn run_diff_test(filename: &str, filter: &FstFilter) {
    // open file with FST library from GTKWave
    let c_path = CString::new(filename).unwrap();
    let exp_handle = unsafe { fst_sys::fstReaderOpen(c_path.as_ptr()) };

    // open file with our library
    let our_f = File::open(filename).expect(&format!("Failed to open {}", filename));
    let mut our_reader = FstReader::open(our_f).unwrap();

    // compare header
    let exp_header = fst_sys_load_header(exp_handle);
    let our_header = our_reader.get_header();
    assert_eq!(our_header, exp_header);

    // close C-library handle
    unsafe { fst_sys::fstReaderClose(exp_handle) };
}

#[test]
fn diff_verilator_basic_tests_anon() {
    run_diff_test("fsts/VerilatorBasicTests_Anon.fst", &FstFilter::all());
}

#[test]
fn diff_des() {
    run_diff_test("fsts/des.fst", &FstFilter::all());
}

#[test]
fn diff_transaction() {
    run_diff_test("fsts/transaction.fst", &FstFilter::all());
}
