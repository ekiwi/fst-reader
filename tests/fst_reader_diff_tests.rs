// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use fst_native::*;
use std::collections::VecDeque;
use std::ffi::{c_char, c_uchar, c_void, CStr, CString};
use std::fs::File;

mod utils;
use utils::hierarchy_to_str;

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
            timescale_exponent: fst_sys::fstReaderGetTimescale(handle),
        }
    }
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
        fst_sys::fstScopeType_FST_ST_VCD_TASK => "Task",
        fst_sys::fstScopeType_FST_ST_VCD_FUNCTION => "Function",
        fst_sys::fstScopeType_FST_ST_VCD_BEGIN => "Begin",
        fst_sys::fstScopeType_FST_ST_VCD_FORK => "Fork",
        fst_sys::fstScopeType_FST_ST_VCD_GENERATE => "Generate",
        fst_sys::fstScopeType_FST_ST_VCD_STRUCT => "Struct",
        fst_sys::fstScopeType_FST_ST_VCD_UNION => "Union",
        fst_sys::fstScopeType_FST_ST_VCD_CLASS => "Class",
        fst_sys::fstScopeType_FST_ST_VCD_INTERFACE => "Interface",
        fst_sys::fstScopeType_FST_ST_VCD_PACKAGE => "Package",
        fst_sys::fstScopeType_FST_ST_VCD_PROGRAM => "Program",
        fst_sys::fstScopeType_FST_ST_VHDL_ARCHITECTURE => "VhdlArchitecture",
        fst_sys::fstScopeType_FST_ST_VHDL_PROCEDURE => "VhdlProcedure",
        fst_sys::fstScopeType_FST_ST_VHDL_FUNCTION => "VhdlFunction",
        fst_sys::fstScopeType_FST_ST_VHDL_RECORD => "VhdlRecord",
        fst_sys::fstScopeType_FST_ST_VHDL_PROCESS => "VhdlProcess",
        fst_sys::fstScopeType_FST_ST_VHDL_BLOCK => "VhdlBlock",
        fst_sys::fstScopeType_FST_ST_VHDL_FOR_GENERATE => "VhdlForGenerate",
        fst_sys::fstScopeType_FST_ST_VHDL_IF_GENERATE => "VhdlIfGenerate",
        fst_sys::fstScopeType_FST_ST_VHDL_GENERATE => "VhdlGenerate",
        fst_sys::fstScopeType_FST_ST_VHDL_PACKAGE => "VhdlPackage",
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
                7 => {
                    // FST_MT_ENUMTABLE (missing from fst_sys)
                    if name.is_empty() {
                        format!("EnumTableRef: {}", attr.arg)
                    } else {
                        format!("EnumTable: {name} ({})", attr.arg)
                    }
                }
                fst_sys::fstMiscType_FST_MT_COMMENT => {
                    format!("Comment: {name}")
                }
                fst_sys::fstMiscType_FST_MT_SUPVAR => {
                    let type_name = name;
                    let svt = attr.arg >> fst_sys::fstSupplementalDataType_FST_SDT_SVT_SHIFT_COUNT;
                    let sdt = attr.arg & (fst_sys::fstSupplementalDataType_FST_SDT_ABS_MAX as u64);
                    format!(
                        "VHDL Var Info: {type_name}, {}, {}",
                        fst_sys_svt_to_str(svt),
                        fst_sys_sdt_to_str(sdt)
                    )
                }
                other => todo!("misc attribute of subtype {other}"),
            }
        }
        _ => format!("BeginAttr: {name}"),
    }
}

fn fst_sys_svt_to_str(svt: u64) -> &'static str {
    match svt as fst_sys::fstSupplementalVarType {
        fst_sys::fstSupplementalVarType_FST_SVT_NONE => "None",
        fst_sys::fstSupplementalVarType_FST_SVT_VHDL_SIGNAL => "Signal",
        fst_sys::fstSupplementalVarType_FST_SVT_VHDL_VARIABLE => "Variable",
        fst_sys::fstSupplementalVarType_FST_SVT_VHDL_CONSTANT => "Constant",
        fst_sys::fstSupplementalVarType_FST_SVT_VHDL_FILE => "File",
        fst_sys::fstSupplementalVarType_FST_SVT_VHDL_MEMORY => "Memory",
        _ => "INVALID",
    }
}

fn fst_sys_sdt_to_str(sdt: u64) -> &'static str {
    match sdt as fst_sys::fstSupplementalDataType {
        fst_sys::fstSupplementalDataType_FST_SDT_NONE => "None",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_BOOLEAN => "Boolean",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_BIT => "Bit",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_BIT_VECTOR => "Vector",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_STD_ULOGIC => "ULogic",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_STD_ULOGIC_VECTOR => "ULogicVector",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_STD_LOGIC => "Logic",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_STD_LOGIC_VECTOR => "LogicVector",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_UNSIGNED => "Unsigned",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_SIGNED => "Signed",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_INTEGER => "Integer",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_REAL => "Real",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_NATURAL => "Natural",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_POSITIVE => "Positive",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_TIME => "Time",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_CHARACTER => "Character",
        fst_sys::fstSupplementalDataType_FST_SDT_VHDL_STRING => "String",
        _ => "INVALID",
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

fn diff_hierarchy<R: std::io::BufRead + std::io::Seek>(
    our_reader: &mut FstReader<R>,
    mut exp_hierarchy: VecDeque<String>,
) -> Vec<bool> {
    let mut is_real = Vec::new();
    let check = |entry: FstHierarchyEntry| {
        // remember if variables are real valued
        if let FstHierarchyEntry::Var { tpe, handle, .. } = &entry {
            let is_var_real = matches!(
                tpe,
                FstVarType::Real
                    | FstVarType::RealParameter
                    | FstVarType::RealTime
                    | FstVarType::ShortReal
            );
            let idx = handle.get_index();
            if is_real.len() <= idx {
                is_real.resize(idx + 1, false);
            }
            is_real[idx] = is_var_real;
        };

        let expected = exp_hierarchy.pop_front().unwrap();
        let actual = hierarchy_to_str(&entry);
        assert_eq!(actual, expected);
        // println!("{actual:?}");
    };
    our_reader.read_hierarchy(check).unwrap();
    is_real
}

fn fst_sys_load_signals(handle: *mut c_void, is_real: &[bool]) -> VecDeque<(u64, u32, String)> {
    let mut data = CallbackData {
        out: VecDeque::new(),
        is_real: Vec::from(is_real),
    };
    unsafe {
        fst_sys::fstReaderIterBlocksSetNativeDoublesOnCallback(handle, 1);
        fst_sys::fstReaderSetFacProcessMaskAll(handle);
        let data_ptr = (&mut data) as *mut CallbackData;
        let user_ptr = data_ptr as *mut c_void;
        fst_sys::fstReaderIterBlocks2(
            handle,
            Some(signal_change_callback),
            Some(var_signal_change_callback),
            user_ptr,
            std::ptr::null_mut(),
        );
    }
    data.out
}

struct CallbackData {
    out: VecDeque<(u64, u32, String)>,
    is_real: Vec<bool>,
}

extern "C" fn signal_change_callback(
    data_ptr: *mut c_void,
    time: u64,
    handle: fst_sys::fstHandle,
    value: *const c_uchar,
) {
    let data = unsafe { &mut *(data_ptr as *mut CallbackData) };
    let signal_idx = (handle - 1) as usize;
    let string = if data.is_real[signal_idx] {
        let slic = unsafe { std::slice::from_raw_parts(value as *const u8, 8) };
        let value = f64::from_le_bytes(slic.try_into().unwrap());
        format!("{value}")
    } else {
        unsafe {
            CStr::from_ptr(value as *const c_char)
                .to_str()
                .unwrap()
                .to_string()
        }
    };
    let signal = (time, handle, string);

    data.out.push_back(signal);
}

extern "C" fn var_signal_change_callback(
    data_ptr: *mut c_void,
    time: u64,
    handle: fst_sys::fstHandle,
    value: *const c_uchar,
    len: u32,
) {
    let bytes = unsafe { std::slice::from_raw_parts(value, len as usize) };
    let string: String = std::str::from_utf8(bytes).unwrap().to_string();
    let signal = (time, handle, string);
    let data = unsafe { &mut *(data_ptr as *mut CallbackData) };
    let signal_idx = (handle - 1) as usize;
    assert!(
        !data.is_real[signal_idx],
        "reals should never be variable length!"
    );
    data.out.push_back(signal);
}

fn diff_signals<R: std::io::BufRead + std::io::Seek>(
    our_reader: &mut FstReader<R>,
    mut exp_signals: VecDeque<(u64, u32, String)>,
) {
    let check = |time: u64, handle: FstSignalHandle, value: FstSignalValue| {
        let (exp_time, exp_handle, exp_value) = exp_signals.pop_front().unwrap();
        let actual_as_string = match value {
            FstSignalValue::String(value) => String::from_utf8_lossy(value).to_string(),
            FstSignalValue::Real(value) => format!("{value}"),
        };
        let actual = (time, handle.get_index() + 1, actual_as_string);
        let expected = (exp_time, exp_handle as usize, exp_value);
        assert_eq!(actual, expected);
        // println!("{actual:?}");
    };
    let filter = FstFilter::all();
    our_reader.read_signals(&filter, check).unwrap();
}

fn run_diff_test(filename: &str, _filter: &FstFilter) {
    // open file with FST library from GTKWave
    let c_path = CString::new(filename).unwrap();
    let exp_handle = unsafe { fst_sys::fstReaderOpen(c_path.as_ptr()) };

    // open file with our library
    let our_f = File::open(filename).unwrap_or_else(|_| panic!("Failed to open {}", filename));
    let mut our_reader = FstReader::open(std::io::BufReader::new(our_f)).unwrap();

    // compare header
    let exp_header = fst_sys_load_header(exp_handle);
    let our_header = our_reader.get_header();
    assert_eq!(our_header, exp_header);

    // compare hierarchy
    let exp_hierarchy = fst_sys_load_hierarchy(exp_handle);
    let is_real = diff_hierarchy(&mut our_reader, exp_hierarchy);

    // compare signals
    let exp_signals = fst_sys_load_signals(exp_handle, &is_real);
    diff_signals(&mut our_reader, exp_signals);

    // close C-library handle
    unsafe { fst_sys::fstReaderClose(exp_handle) };
}

#[test]
fn diff_aldec_spi_write() {
    run_diff_test("fsts/aldec/SPI_Write.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_amaranth_up_counter() {
    run_diff_test("fsts/amaranth/up_counter.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_ghdl_alu() {
    run_diff_test("fsts/ghdl/alu.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_ghdl_idea() {
    run_diff_test("fsts/ghdl/idea.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_ghdl_pcpu() {
    run_diff_test("fsts/ghdl/pcpu.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_ghdl_oscar_ghdl() {
    run_diff_test("fsts/ghdl/oscar/ghdl.fst", &FstFilter::all());
}

#[test]
fn diff_ghdl_oscar_vhdl3() {
    run_diff_test("fsts/ghdl/oscar/vhdl3.fst", &FstFilter::all());
}

#[test]
fn diff_gtkwave_des() {
    run_diff_test("fsts/gtkwave-analyzer/des.fst", &FstFilter::all());
}

#[test]
fn diff_gtkwave_perm_current() {
    run_diff_test(
        "fsts/gtkwave-analyzer/perm_current.vcd.fst",
        &FstFilter::all(),
    );
}

#[test]
fn diff_gtkwave_transaction() {
    run_diff_test("fsts/gtkwave-analyzer/transaction.fst", &FstFilter::all());
}

#[test]
fn diff_icarus_cpu() {
    run_diff_test("fsts/icarus/CPU.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_icarus_rv32_soc_tb() {
    run_diff_test("fsts/icarus/rv32_soc_TB.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_icarus_test1() {
    run_diff_test("fsts/icarus/test1.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_model_sim_clkdiv2n_tb() {
    run_diff_test("fsts/model-sim/clkdiv2n_tb.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_model_sim_cpu_design() {
    run_diff_test("fsts/model-sim/CPU_Design.msim.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_my_hdl_sigmoid_tb() {
    run_diff_test("fsts/my-hdl/sigmoid_tb.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_my_hdl_simple_memory() {
    run_diff_test("fsts/my-hdl/Simple_Memory.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_my_hdl_top() {
    run_diff_test("fsts/my-hdl/top.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_ncsim_ffdiv() {
    run_diff_test("fsts/ncsim/ffdiv_32bit_tb.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_quartus_mips_hardware() {
    run_diff_test("fsts/quartus/mipsHardware.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_quartus_wave() {
    run_diff_test("fsts/quartus/wave_registradores.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_questa_sim_dump() {
    run_diff_test("fsts/questa-sim/dump.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_questa_sim_test() {
    run_diff_test("fsts/questa-sim/test.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_riviera_pro_dump() {
    run_diff_test("fsts/riviera-pro/dump.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_systemc_waveform() {
    run_diff_test("fsts/systemc/waveform.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_systemc_waveform_dual_lz4() {
    // To generate this file with a dual packed hierarchy,
    // we used a modified version of vcd2fst with a lower FST_HDR_FOURPACK_DUO_SIZE threshold.
    run_diff_test("fsts/systemc/waveform.vcd.dual_lz4.fst", &FstFilter::all());
}

#[test]
fn diff_systemc_waveform_fastlz() {
    // Generated by passing the `-F` flag to vcd2fst.
    run_diff_test("fsts/systemc/waveform.vcd.fastlz.fst", &FstFilter::all());
}

#[test]
fn diff_systemc_waveform_fastlz_lvl2() {
    // Generated by passing the `-F` flag to vcd2fst and changing the fastlz library to always
    // emit level2.
    run_diff_test(
        "fsts/systemc/waveform.vcd.fastlz_lvl2.fst",
        &FstFilter::all(),
    );
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("fsts/treadle/GCD.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_vcs_apb_uvm_new() {
    run_diff_test("fsts/vcs/Apb_slave_uvm_new.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_vcs_datapath_log() {
    run_diff_test("fsts/vcs/datapath_log.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_vcs_processor() {
    run_diff_test("fsts/vcs/processor.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_verilator_basic_test() {
    run_diff_test("fsts/verilator/basic_test.fst", &FstFilter::all());
}

#[test]
fn diff_verilator_many_sv_data_types() {
    run_diff_test("fsts/verilator/many_sv_datatypes.fst", &FstFilter::all());
}

// FST reported in https://gitlab.com/surfer-project/surfer/-/issues/201
#[test]
fn diff_verilator_surfer_issue_201() {
    run_diff_test("fsts/verilator/surfer_issue_201.fst", &FstFilter::all());
}

#[test]
fn diff_verilator_swerv1() {
    run_diff_test("fsts/verilator/swerv1.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_verilator_vlt_dump() {
    run_diff_test("fsts/verilator/vlt_dump.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_vivado_iladata() {
    run_diff_test("fsts/vivado/iladata.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_xilinx_isim_test() {
    run_diff_test("fsts/xilinx_isim/test.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_xilinx_isim_test1() {
    run_diff_test("fsts/xilinx_isim/test1.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_xilinx_isim_test2x2_regex22_string1() {
    run_diff_test(
        "fsts/xilinx_isim/test2x2_regex22_string1.vcd.fst",
        &FstFilter::all(),
    );
}

#[test]
fn diff_scope_with_comment() {
    run_diff_test("fsts/scope_with_comment.vcd.fst", &FstFilter::all());
}

#[test]
fn diff_vcd_file_with_errors() {
    run_diff_test("fsts/VCD_file_with_errors.vcd.fst", &FstFilter::all());
}
