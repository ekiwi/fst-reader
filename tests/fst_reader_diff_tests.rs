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

fn fst_sys_load_hierarchy(handle: *mut c_void) -> VecDeque<String> {
    let mut out = VecDeque::new();
    unsafe { fst_sys::fstReaderIterateHierRewind(handle) };
    loop {
        let p = unsafe {
            let ptr = fst_sys::fstReaderIterateHier(handle);
            if ptr.is_null() {
                None
            } else {
                Some(&*ptr)
            }
        };
        if p.is_none() {
            break;
        }
        let value = p.unwrap();
        out.push_back(fst_sys_hierarchy_to_str(value));
    }
    out
}

unsafe fn fst_sys_hierarchy_read_name(ptr: *const c_char, len: u32) -> String {
    let slic = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    (std::str::from_utf8(slic)).unwrap().to_string()
}

fn fst_sys_scope_tpe_to_string(tpe: fst_sys::fstScopeType) -> String {
    let con = match tpe {
        fst_sys::fstScopeType_FST_ST_VCD_MODULE => "Module",
        other => todo!("scope type: {other}"),
    };
    con.to_string()
}

unsafe fn fst_sys_parse_attribute(attr: &fst_sys::fstHier__bindgen_ty_1_fstHierAttr) -> String {
    let name = fst_sys_hierarchy_read_name(attr.name, attr.name_length);
    match attr.typ as fst_sys::fstAttrType {
        fst_sys::fstAttrType_FST_AT_MISC => {
            let misc_tpe = attr.subtype as fst_sys::fstMiscType;
            match misc_tpe {
                fst_sys::fstMiscType_FST_MT_PATHNAME => {
                    let id = attr.arg;
                    format!("PathName: {id} -> {name}")
                }
                fst_sys::fstMiscType_FST_MT_SOURCEISTEM
                | fst_sys::fstMiscType_FST_MT_SOURCESTEM => {
                    let line = attr.arg;
                    let path_id = leb128::read::unsigned(&mut name.as_bytes()).unwrap();
                    let is_instantiation = misc_tpe == fst_sys::fstMiscType_FST_MT_SOURCEISTEM;
                    format!("SourceStem:: {is_instantiation}, {path_id}, {line}")
                }
                other => todo!("misc attribute of subtype {other}"),
            }
        }
        _ => format!("BeginAttr: {name}"),
    }
}

fn fst_sys_hierarchy_to_str(entry: &fst_sys::fstHier) -> String {
    match entry.htyp as u32 {
        fst_sys::fstHierType_FST_HT_SCOPE => {
            let name = unsafe {
                fst_sys_hierarchy_read_name(entry.u.scope.name, entry.u.scope.name_length)
            };
            let component = unsafe {
                fst_sys_hierarchy_read_name(entry.u.scope.component, entry.u.scope.component_length)
            };
            let tpe =
                unsafe { fst_sys_scope_tpe_to_string(entry.u.scope.typ as fst_sys::fstScopeType) };
            format!("Scope: {name} ({tpe}) {component}")
        }
        fst_sys::fstHierType_FST_HT_UPSCOPE => "UpScope".to_string(),
        fst_sys::fstHierType_FST_HT_VAR => {
            let handle = unsafe { entry.u.var.handle };
            let name =
                unsafe { fst_sys_hierarchy_read_name(entry.u.var.name, entry.u.var.name_length) };
            format!("(H{handle}): {name}")
        }
        fst_sys::fstHierType_FST_HT_ATTRBEGIN => unsafe { fst_sys_parse_attribute(&entry.u.attr) },
        fst_sys::fstHierType_FST_HT_ATTREND => "EndAttr".to_string(),
        other => todo!("htype={other}"),
    }
}

fn hierarchy_to_str(entry: &FstHierarchyEntry) -> String {
    match entry {
        FstHierarchyEntry::Scope {
            name,
            tpe,
            component,
        } => format!("Scope: {name} ({}) {component}", hierarchy_tpe_to_str(tpe)),
        FstHierarchyEntry::UpScope => "UpScope".to_string(),
        FstHierarchyEntry::Var { name, handle, .. } => format!("({handle}): {name}"),
        FstHierarchyEntry::AttributeBegin { name } => format!("BeginAttr: {name}"),
        FstHierarchyEntry::AttributeEnd => "EndAttr".to_string(),
        FstHierarchyEntry::PathName { name, id } => format!("PathName: {id} -> {name}"),
        FstHierarchyEntry::SourceStem {
            is_instantiation,
            path_id,
            line,
        } => format!("SourceStem:: {is_instantiation}, {path_id}, {line}"),
    }
}

fn hierarchy_tpe_to_str(tpe: &FstScopeType) -> String {
    let con = match tpe {
        FstScopeType::Module => "Module",
        FstScopeType::Task => "Task",
        FstScopeType::Function => "Function",
        FstScopeType::Begin => "Begin",
        FstScopeType::Fork => "Fork",
        FstScopeType::Generate => "Generate",
        FstScopeType::Struct => "Struct",
        FstScopeType::Union => "Union",
        FstScopeType::Class => "Class",
        FstScopeType::Interface => "Interface",
        FstScopeType::Package => "Package",
        FstScopeType::Program => "Program",
        FstScopeType::VhdlArchitecture => "VhdlArchitecture",
        FstScopeType::VhdlProcedure => "VhdlProcedure",
        FstScopeType::VhdlFunction => "VhdlFunction",
        FstScopeType::VhdlRecord => "VhdlRecord",
        FstScopeType::VhdlProcess => "VhdlProcess",
        FstScopeType::VhdlBlock => "VhdlBlock",
        FstScopeType::VhdlForGenerate => "VhdlForGenerate",
        FstScopeType::VhdlIfGenerate => "VhdlIfGenerate",
        FstScopeType::VhdlGenerate => "VhdlGenerate",
        FstScopeType::VhdlPackage => "VhdlPackage",
        FstScopeType::AttributeBegin => "AttributeBegin",
        FstScopeType::AttributeEnd => "AttributeEnd",
        FstScopeType::VcdScope => "VcdScope",
        FstScopeType::VcdUpScope => "VcdUpScope",
    };
    con.to_string()
}

fn diff_hierarchy<R: std::io::Read + std::io::Seek>(
    our_reader: &mut FstReader<R>,
    mut exp_hierarchy: VecDeque<String>,
) {
    let check = |entry: FstHierarchyEntry| {
        let expected = exp_hierarchy.pop_front().unwrap();
        let actual = hierarchy_to_str(&entry);
        assert_eq!(actual, expected);
        // println!("{actual:?}");
    };
    our_reader.read_hierarchy(check).unwrap();
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
        // println!("{actual:?}");
    };
    let filter = FstFilter::all();
    our_reader.read_signals(&filter, check).unwrap();
}

fn run_diff_test(filename: &str, filter: &FstFilter) {
    // open file with FST library from GTKWave
    let c_path = CString::new(filename).unwrap();
    let exp_handle = unsafe { fst_sys::fstReaderOpen(c_path.as_ptr()) };

    // open file with our library
    let our_f = File::open(filename).unwrap_or_else(|_| panic!("Failed to open {}", filename));
    let mut our_reader = FstReader::open(our_f).unwrap();

    // compare header
    let exp_header = fst_sys_load_header(exp_handle);
    let our_header = our_reader.get_header();
    assert_eq!(our_header, exp_header);

    // compare hierarchy
    let exp_hierarchy = fst_sys_load_hierarchy(exp_handle);
    diff_hierarchy(&mut our_reader, exp_hierarchy);

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
fn diff_des() {
    run_diff_test("fsts/des.fst", &FstFilter::all());
}

#[test]
fn diff_transaction() {
    run_diff_test("fsts/transaction.fst", &FstFilter::all());
}
