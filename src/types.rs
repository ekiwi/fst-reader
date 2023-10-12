// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
// Contains FST in-memory types.

use bitvec::order::Msb0;
use bitvec::prelude::BitVec;
use num_enum::TryFromPrimitive;
use std::fmt::Formatter;
use std::num::NonZeroU32;

pub(crate) const HIERARCHY_NAME_MAX_SIZE: usize = 512;
pub(crate) const HIERARCHY_ATTRIBUTE_MAX_SIZE: usize = 65536 + 4096;

#[derive(Debug)]
pub struct FstSignalHandle(NonZeroU32);

impl FstSignalHandle {
    pub(crate) fn new(value: u32) -> Self {
        FstSignalHandle(NonZeroU32::new(value).unwrap())
    }
    pub(crate) fn from_index(index: usize) -> Self {
        FstSignalHandle(NonZeroU32::new((index as u32) + 1).unwrap())
    }
    pub fn get_index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl std::fmt::Display for FstSignalHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "H{}", self.0)
    }
}

#[derive(Debug)]
pub(crate) struct EnumHandle(u32);

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum FloatingPointEndian {
    Little,
    Big,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub(crate) enum FileType {
    Verilog = 0,
    Vhdl = 1,
    VerilogVhdl = 2,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, PartialEq)]
pub(crate) enum BlockType {
    Header = 0,
    VcData = 1,
    Blackout = 2,
    Geometry = 3,
    Hierarchy = 4,
    VcDataDynamicAlias = 5,
    HierarchyLZ4 = 6,
    HierarchyLZ4Duo = 7,
    VcDataDynamicAlias2 = 8,
    GZipWrapper = 254,
    Skip = 255,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum FstScopeType {
    // VCD
    Module = 0,
    Task = 1,
    Function = 2,
    Begin = 3,
    Fork = 4,
    Generate = 5,
    Struct = 6,
    Union = 7,
    Class = 8,
    Interface = 9,
    Package = 10,
    Program = 11,
    // VHDL
    VhdlArchitecture = 12,
    VhdlProcedure = 13,
    VhdlFunction = 14,
    VhdlRecord = 15,
    VhdlProcess = 16,
    VhdlBlock = 17,
    VhdlForGenerate = 18,
    VhdlIfGenerate = 19,
    VhdlGenerate = 20,
    VhdlPackage = 21,
    //
    AttributeBegin = 252,
    AttributeEnd = 253,
    //
    VcdScope = 254,
    VcdUpScope = 255,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, PartialEq, Copy, Clone)]
pub enum FstVarType {
    // VCD
    Event = 0,
    Integer = 1,
    Parameter = 2,
    Real = 3,
    RealParameter = 4,
    Reg = 5,
    Supply0 = 6,
    Supply1 = 7,
    Time = 8,
    Tri = 9,
    TriAnd = 10,
    TriOr = 11,
    TriReg = 12,
    Tri0 = 13,
    Tri1 = 14,
    Wand = 15, // or WAnd ?
    Wire = 16,
    Wor = 17, // or WOr?
    Port = 18,
    SparseArray = 19,
    RealTime = 20,
    GenericString = 21,
    // System Verilog
    Bit = 22,
    Logic = 23,
    Int = 24,
    ShortInt = 25,
    LongInt = 26,
    Byte = 27,
    Enum = 28,
    ShortReal = 29,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum FstVarDirection {
    Implicit = 0,
    Input = 1,
    Output = 2,
    InOut = 3,
    Buffer = 4,
    Linkage = 5,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, PartialEq)]
pub(crate) enum AttributeType {
    Misc = 0,
    Array = 1,
    Enum = 2,
    Pack = 3,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, PartialEq)]
pub(crate) enum MiscType {
    Comment = 0,
    EnvVar = 1,
    SupVar = 2,
    PathName = 3,
    SourceStem = 4,
    SourceInstantiationStem = 5,
    ValueList = 6,
    EnumTable = 7,
    Unknown = 8,
}

pub(crate) const DOUBLE_ENDIAN_TEST: f64 = std::f64::consts::E;

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct Header {
    pub(crate) start_time: u64,
    pub(crate) end_time: u64,
    pub(crate) memory_used_by_writer: u64,
    pub(crate) scope_count: u64,
    pub(crate) var_count: u64,
    pub(crate) max_var_id_code: u64, // aka maxhandle
    pub(crate) vc_section_count: u64,
    pub(crate) timescale_exponent: i8,
    pub(crate) version: String,
    pub(crate) date: String,
    pub(crate) file_type: FileType,
    pub(crate) time_zero: u64,
    pub(crate) float_endian: FloatingPointEndian,
}

#[derive(Debug)]
pub(crate) struct Signals {
    // called "geometry" in gtkwave
    pub(crate) lengths: Vec<u32>,
    pub(crate) types: Vec<FstVarType>,
}

#[derive(Debug, Clone)]
pub(crate) struct DataSectionInfo {
    pub(crate) file_offset: u64, // points to section length
    pub(crate) start_time: u64,
    pub(crate) end_time: u64,
    pub(crate) kind: DataSectionKind,
}

#[derive(Debug)]
pub enum FstHierarchyEntry {
    Scope {
        tpe: FstScopeType,
        name: String,
        component: String,
    },
    UpScope,
    Var {
        tpe: FstVarType,
        direction: FstVarDirection,
        name: String,
        length: u32,
        handle: FstSignalHandle,
        is_alias: bool,
    },
    AttributeBegin {
        name: String,
        // TODO
    },
    PathName {
        /// this id is used by other attributes to refer to the path
        id: u64,
        name: String,
    },
    SourceStem {
        is_instantiation: bool,
        path_id: u64,
        line: u64,
    },
    AttributeEnd,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum HierarchyCompression {
    ZLib,
    Lz4,
    Lz4Duo,
}

pub(crate) type BitMask = BitVec<u8, Msb0>;

pub(crate) struct DataFilter {
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) signals: BitMask,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DataSectionKind {
    Standard,
    DynamicAlias,
    DynamicAlias2,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ValueChangePackType {
    Lz4,
    FastLz,
    Zlib,
}

impl ValueChangePackType {
    pub(crate) fn from_u8(value: u8) -> Self {
        match value {
            b'4' => ValueChangePackType::Lz4,
            b'F' => ValueChangePackType::FastLz,
            _ => ValueChangePackType::Zlib,
        }
    }
}

impl DataSectionKind {
    pub(crate) fn from_block_type(tpe: &BlockType) -> Option<Self> {
        match tpe {
            BlockType::VcData => Some(DataSectionKind::Standard),
            BlockType::VcDataDynamicAlias => Some(DataSectionKind::DynamicAlias),
            BlockType::VcDataDynamicAlias2 => Some(DataSectionKind::DynamicAlias2),
            _ => None,
        }
    }
}
