// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use fst_native::*;
use std::collections::VecDeque;
use std::ffi::{c_char, c_uchar, c_void, CStr, CString};
use std::fs::File;

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

// imported from fst_sys
type FstChangeCallback = extern "C" fn(*mut c_void, u64, fst_sys::fstHandle, *const c_uchar);
unsafe fn unpack_closure<F>(closure: &mut F) -> (*mut c_void, FstChangeCallback)
where
    F: FnMut(u64, fst_sys::fstHandle, *const c_uchar),
{
    extern "C" fn trampoline<F>(
        data: *mut c_void,
        time: u64,
        handle: fst_sys::fstHandle,
        value: *const c_uchar,
    ) where
        F: FnMut(u64, fst_sys::fstHandle, *const c_uchar),
    {
        let closure: &mut F = unsafe { &mut *(data as *mut F) };
        (*closure)(time, handle, value);
    }
    (closure as *mut F as *mut c_void, trampoline::<F>)
}

fn fst_sys_load_signals(handle: *mut c_void) -> VecDeque<(u64, u32, String)> {
    let mut out = VecDeque::new();
    let mut f = |time: u64, handle: fst_sys::fstHandle, value: *const c_uchar| {
        let string: String = unsafe {
            CStr::from_ptr(value as *const c_char)
                .to_str()
                .unwrap()
                .to_string()
        };
        out.push_back((time, handle, string));
    };
    unsafe {
        fst_sys::fstReaderSetFacProcessMaskAll(handle);
        let (data, f) = unpack_closure(&mut f);
        fst_sys::fstReaderIterBlocks(handle, Some(f), data, std::ptr::null_mut());
    }
    out
}

fn diff_signals<R: std::io::Read + std::io::Seek>(
    our_reader: &mut FstReader<R>,
    mut exp_signals: VecDeque<(u64, u32, String)>,
) {
    let check = |time: u64, handle: FstSignalHandle, value: &str| {
        let (exp_time, exp_handle, exp_value) = exp_signals.pop_front().unwrap();
        let actual = (time, handle.get_index() + 1, value);
        let expected = (exp_time, exp_handle as usize, exp_value.as_str());
        assert_eq!(actual, expected);
    };
    let filter = FstFilter::all();
    our_reader.read_signals(&filter, check).unwrap();
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

    // compare signals
    let exp_signals = fst_sys_load_signals(exp_handle);
    diff_signals(&mut our_reader, exp_signals);

    // close C-library handle
    unsafe { fst_sys::fstReaderClose(exp_handle) };
}

#[test]
fn diff_verilator_basic_tests_anon() {
    run_diff_test("fsts/VerilatorBasicTests_Anon.fst", &FstFilter::all());
}

#[test]
#[ignore]
fn diff_des() {
    run_diff_test("fsts/des.fst", &FstFilter::all());
}

#[test]
fn diff_transaction() {
    run_diff_test("fsts/transaction.fst", &FstFilter::all());
}
