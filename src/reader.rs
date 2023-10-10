// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use bitvec::prelude::{BitVec, Msb0};
use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::fmt::Formatter;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU32;

/// Reads in a FST file.
pub struct FstReader<R: Read + Seek> {
    input: InputVariant<R>,
    meta: MetaData,
}

enum InputVariant<R: Read + Seek> {
    Original(R),
    Uncompressed(BufReader<std::fs::File>),
}

pub struct FstFilter {
    pub start: u64,
    pub end: Option<u64>,
    pub include: Option<Vec<FstSignalHandle>>,
}

impl FstFilter {
    pub fn all() -> Self {
        FstFilter {
            start: 0,
            end: None,
            include: None,
        }
    }

    pub fn new(start: u64, end: u64, signals: Vec<FstSignalHandle>) -> Self {
        FstFilter {
            start,
            end: Some(end),
            include: Some(signals),
        }
    }

    pub fn filter_time(start: u64, end: u64) -> Self {
        FstFilter {
            start,
            end: Some(end),
            include: None,
        }
    }

    pub fn filter_signals(signals: Vec<FstSignalHandle>) -> Self {
        FstFilter {
            start: 0,
            end: None,
            include: Some(signals),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FstHeader {
    /// time of first sample
    pub start_time: u64,
    /// time of last sample
    pub end_time: u64,
    /// number of variables in the design
    pub var_count: u64,
    /// the highest signal handle; indicates the number of unique signals
    pub max_handle: u64,
    /// human readable version string
    pub version: String,
    /// human readable times stamp
    pub date: String,
}

impl<R: Read + Seek> FstReader<R> {
    /// Reads in the FST file meta-data.
    pub fn open(mut input: R) -> Result<Self> {
        let uncompressed_input = uncompress_gzip_wrapper(&mut input)?;
        match uncompressed_input {
            None => {
                let mut header_reader = HeaderReader::new(input);
                header_reader.read()?;
                let (input, meta) = header_reader.into_input_and_meta_data().unwrap();
                Ok(FstReader {
                    input: InputVariant::Original(input),
                    meta,
                })
            }
            Some(uc) => {
                let mut header_reader = HeaderReader::new(uc);
                header_reader.read()?;
                let (uc2, meta) = header_reader.into_input_and_meta_data().unwrap();
                Ok(FstReader {
                    input: InputVariant::Uncompressed(uc2),
                    meta,
                })
            }
        }
    }

    pub fn get_header(&self) -> FstHeader {
        FstHeader {
            start_time: self.meta.header.start_time,
            end_time: self.meta.header.end_time,
            var_count: self.meta.header.var_count,
            max_handle: self.meta.header.max_var_id_code,
            version: self.meta.header.version.clone(),
            date: self.meta.header.date.clone(),
        }
    }

    /// Reads the hierarchy and calls callback for every item.
    pub fn read_hierarchy(&mut self, callback: impl FnMut(FstHierarchyEntry)) -> Result<()> {
        match &mut self.input {
            InputVariant::Original(input) => read_hierarchy(input, &self.meta, callback),
            InputVariant::Uncompressed(input) => read_hierarchy(input, &self.meta, callback),
        }
    }

    /// Read signal values for a specific time interval.
    pub fn read_signals(
        &mut self,
        filter: &FstFilter,
        callback: impl FnMut(u64, FstSignalHandle, &str),
    ) -> Result<()> {
        // convert user filters
        let signal_count = self.meta.signals.types.len();
        let signal_mask = if let Some(signals) = &filter.include {
            let mut signal_mask = BitMask::repeat(false, signal_count);
            for sig in signals {
                let signal_idx = (sig.0.get() - 1) as usize;
                signal_mask.set(signal_idx, true);
            }
            signal_mask
        } else {
            // include all
            BitMask::repeat(true, signal_count)
        };
        let data_filter = DataFilter {
            start: filter.start,
            end: filter.end.unwrap_or(self.meta.header.end_time),
            signals: signal_mask,
        };

        // build and run reader
        match &mut self.input {
            InputVariant::Original(input) => {
                read_signals(input, &self.meta, &data_filter, callback)
            }
            InputVariant::Uncompressed(input) => {
                read_signals(input, &self.meta, &data_filter, callback)
            }
        }
    }
}

fn read_hierarchy(
    input: &mut (impl Read + Seek),
    meta: &MetaData,
    mut callback: impl FnMut(FstHierarchyEntry),
) -> Result<()> {
    input.seek(SeekFrom::Start(meta.hierarchy_offset))?;
    let bytes = read_hierarchy_bytes(input, meta.hierarchy_compression)?;
    let mut input = bytes.as_slice();
    let mut handle_count = 0u32;
    while let Some(entry) = read_hierarchy_entry(&mut input, &mut handle_count)? {
        callback(entry);
    }
    Ok(())
}

fn read_signals(
    input: &mut (impl Read + Seek),
    meta: &MetaData,
    filter: &DataFilter,
    mut callback: impl FnMut(u64, FstSignalHandle, &str),
) -> Result<()> {
    let mut reader = DataReader {
        input,
        meta,
        filter,
        callback: &mut callback,
    };
    reader.read()
}

/// Checks to see if the whole file is compressed in which case it is decompressed
/// to a temp file which is returned.
fn uncompress_gzip_wrapper(
    input: &mut (impl Read + Seek),
) -> Result<Option<BufReader<std::fs::File>>> {
    let block_tpe = read_block_tpe(input)?;
    if block_tpe != BlockType::GZipWrapper {
        // no gzip wrapper
        input.seek(SeekFrom::Start(0))?;
        return Ok(None);
    } else {
        // uncompress
        let section_length = read_u64(input)?;
        let uncompress_length = read_u64(input)? as usize;
        if section_length == 0 {
            let kind = ReaderErrorKind::NotFinishedCompressing();
            return Err(ReaderError { kind });
        }

        let mut target = tempfile::tempfile().unwrap();
        let mut decoder = flate2::read::GzDecoder::new(input);
        let mut buf = vec![0u8; 32768]; // FST_GZIO_LEN
        let mut remaining = uncompress_length;
        while remaining > 0 {
            let read_len = std::cmp::min(buf.len(), remaining);
            remaining -= read_len;
            decoder.read_exact(&mut buf[..read_len])?;
            target.write(&mut buf[..read_len])?;
        }
        // go to start of new file and return
        target.seek(SeekFrom::Start(0))?;
        let new_input = std::io::BufReader::new(target);
        Ok(Some(new_input))
    }
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
enum FileType {
    Verilog = 0,
    Vhdl = 1,
    VerilogVhdl = 2,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, PartialEq)]
enum BlockType {
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
enum AttributeType {
    Misc = 0,
    Array = 1,
    Enum = 2,
    Pack = 3,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, PartialEq)]
enum MiscType {
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

const DOUBLE_ENDIAN_TEST: f64 = 2.7182818284590452354;

#[derive(Debug)]
struct Header {
    start_time: u64,
    end_time: u64,
    memory_used_by_writer: u64,
    scope_count: u64,
    var_count: u64,
    max_var_id_code: u64, // aka maxhandle
    vc_section_count: u64,
    timescale_exponent: i8,
    version: String,
    date: String,
    file_type: FileType,
    time_zero: u64,
}

#[derive(Debug)]
struct Signals {
    // called "geometry" in gtkwave
    lengths: Vec<u32>,
    types: Vec<FstVarType>,
}

#[derive(Debug, Clone)]
struct DataSectionInfo {
    file_offset: u64, // points to section length
    start_time: u64,
    end_time: u64,
    kind: DataSectionKind,
}

#[derive(Debug)]
struct MetaData {
    header: Header,
    signals: Signals,
    data_sections: Vec<DataSectionInfo>,
    float_endian: FloatingPointEndian,
    hierarchy_compression: HierarchyCompression,
    hierarchy_offset: u64,
}

#[derive(Debug)]
pub struct FstSignalHandle(NonZeroU32);
#[derive(Debug)]
struct EnumHandle(u32);

impl FstSignalHandle {
    fn new(value: u32) -> Self {
        FstSignalHandle(NonZeroU32::new(value).unwrap())
    }
    fn from_index(index: usize) -> Self {
        FstSignalHandle(NonZeroU32::new((index as u32) + 1).unwrap())
    }
}

impl std::fmt::Display for FstSignalHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "H{}", self.0)
    }
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
enum HierarchyCompression {
    ZLib,
    Lz4,
    Lz4Duo,
}

#[derive(Debug)]
pub enum ReaderErrorKind {
    IO(std::io::Error),
    FromPrimitive(),
    StringParse(std::str::Utf8Error),
    StringParse2(std::string::FromUtf8Error),
    ParseVariant(),
    DecompressLz4(lz4_flex::block::DecompressError),
    /// The FST file is still being compressed into its final GZIP wrapper.
    NotFinishedCompressing(),
}
#[derive(Debug)]
pub struct ReaderError {
    kind: ReaderErrorKind,
}

impl From<std::io::Error> for ReaderError {
    fn from(value: std::io::Error) -> Self {
        let kind = ReaderErrorKind::IO(value);
        ReaderError { kind }
    }
}

impl<Enum: TryFromPrimitive> From<TryFromPrimitiveError<Enum>> for ReaderError {
    fn from(_value: TryFromPrimitiveError<Enum>) -> Self {
        let kind = ReaderErrorKind::FromPrimitive();
        ReaderError { kind }
    }
}

impl From<std::str::Utf8Error> for ReaderError {
    fn from(value: std::str::Utf8Error) -> Self {
        let kind = ReaderErrorKind::StringParse(value);
        ReaderError { kind }
    }
}

impl From<std::string::FromUtf8Error> for ReaderError {
    fn from(value: std::string::FromUtf8Error) -> Self {
        let kind = ReaderErrorKind::StringParse2(value);
        ReaderError { kind }
    }
}

impl From<lz4_flex::block::DecompressError> for ReaderError {
    fn from(value: lz4_flex::block::DecompressError) -> Self {
        let kind = ReaderErrorKind::DecompressLz4(value);
        ReaderError { kind }
    }
}

// impl From<miniz_oxide::inflate::DecompressError> for ReaderError {
//     fn from(value: miniz_oxide::inflate::DecompressError) -> Self {
//         let kind = ReaderErrorKind::DecompressZLib(value);
//         ReaderError { kind }
//     }
// }

pub type Result<T> = std::result::Result<T, ReaderError>;

#[inline]
fn read_variant_u32(input: &mut impl Read) -> Result<(u32, u32)> {
    let mut byte = [0u8; 1];
    let mut res = 0u32;
    // 32bit / 7bit = ~4.6
    for ii in 0..5u32 {
        input.read_exact(&mut byte)?;
        let value = (byte[0] as u32) & 0x7f;
        res |= value << (7 * ii);
        if (byte[0] & 0x80) == 0 {
            return Ok((res, ii + 1));
        }
    }
    Err(ReaderError {
        kind: ReaderErrorKind::ParseVariant(),
    })
}

#[inline]
fn read_variant_i64(input: &mut impl Read) -> Result<i64> {
    let mut byte = [0u8; 1];
    let mut res = 0i64;
    // 64bit / 7bit = ~9.1
    for ii in 0..10 {
        input.read_exact(&mut byte)?;
        let value = (byte[0] & 0x7f) as i64;
        let shift_by = 7 * ii;
        res |= value << shift_by;
        if (byte[0] & 0x80) == 0 {
            // sign extend
            let sign_bit_set = (byte[0] & 0x40) != 0;
            if shift_by < (8 * 8) && sign_bit_set {
                res |= -(1i64 << (shift_by + 7))
            }
            return Ok(res);
        }
    }
    Err(ReaderError {
        kind: ReaderErrorKind::ParseVariant(),
    })
}

#[inline]
fn read_variant_u64(input: &mut impl Read) -> Result<u64> {
    let mut byte = [0u8; 1];
    let mut res = 0u64;
    for ii in 0..10 {
        // 64bit / 7bit = ~9.1
        input.read_exact(&mut byte)?;
        let value = (byte[0] as u64) & 0x7f;
        res |= value << (7 * ii);
        if (byte[0] & 0x80) == 0 {
            return Ok(res);
        }
    }
    Err(ReaderError {
        kind: ReaderErrorKind::ParseVariant(),
    })
}

#[inline]
fn read_u64(input: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

#[inline]
fn read_u8(input: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
fn read_i8(input: &mut impl Read) -> Result<i8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(i8::from_be_bytes(buf))
}

fn read_c_str(input: &mut impl Read, max_len: usize) -> Result<String> {
    let mut bytes: Vec<u8> = Vec::with_capacity(32);
    for _ in 0..max_len {
        let byte = read_u8(input)?;
        if byte == 0 {
            break;
        } else {
            bytes.push(byte);
        }
    }
    Ok(String::from_utf8(bytes)?)
}

#[inline] // inline to specialize on length
fn read_c_str_fixed_length(input: &mut impl Read, len: usize) -> Result<String> {
    let mut bytes = read_bytes(input, len)?;
    let zero_index = bytes.iter().position(|b| *b == 0u8).unwrap_or(len - 1);
    let str_len = zero_index;
    bytes.truncate(str_len);
    Ok(String::from_utf8(bytes)?)
}

fn read_compressed_bytes(
    input: &mut (impl Read + Seek),
    uncompressed_length: u64,
    compressed_length: u64,
) -> Result<Vec<u8>> {
    let bytes = if uncompressed_length == compressed_length {
        read_bytes(input, compressed_length as usize)?
    } else {
        let start = input.stream_position().unwrap();
        let mut d = flate2::read::ZlibDecoder::new(input);
        let mut uncompressed: Vec<u8> = Vec::with_capacity(uncompressed_length as usize);
        d.read_to_end(&mut uncompressed)?;
        // sanity checks
        assert_eq!(d.total_out(), uncompressed_length);
        assert!(d.total_in() <= compressed_length);
        // the decoder often reads more bytes than it should, we fix this here
        d.into_inner()
            .seek(SeekFrom::Start(start + compressed_length))?;
        uncompressed
    };
    assert_eq!(bytes.len(), uncompressed_length as usize);
    Ok(bytes)
}

fn read_block_tpe(input: &mut impl Read) -> Result<BlockType> {
    Ok(BlockType::try_from(read_u8(input)?)?)
}

#[inline]
fn read_bytes(input: &mut impl Read, len: usize) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    input.take(len as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

#[inline]
fn read_f64(input: &mut impl Read, endian: FloatingPointEndian) -> Result<f64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    match endian {
        FloatingPointEndian::Little => Ok(f64::from_le_bytes(buf)),
        FloatingPointEndian::Big => Ok(f64::from_be_bytes(buf)),
    }
}

const HIERARCHY_NAME_MAX_SIZE: usize = 512;
const HIERARCHY_ATTRIBUTE_MAX_SIZE: usize = 65536 + 4096;

#[derive(Debug, PartialEq, Clone, Copy)]
enum FloatingPointEndian {
    Little,
    Big,
}

fn determine_f64_endian(input: &mut impl Read, needle: f64) -> Result<FloatingPointEndian> {
    let bytes = read_bytes(input, 8)?;
    let mut byte_reader: &[u8] = &bytes;
    let le = read_f64(&mut byte_reader, FloatingPointEndian::Little)?;
    if le == needle {
        return Ok(FloatingPointEndian::Little);
    }
    byte_reader = &bytes;
    let be = read_f64(&mut byte_reader, FloatingPointEndian::Big)?;
    if be == needle {
        Ok(FloatingPointEndian::Big)
    } else {
        todo!("should not get here")
    }
}

struct HeaderReader<R: Read + Seek> {
    input: R,
    header: Option<Header>,
    signals: Option<Signals>,
    data_sections: Vec<DataSectionInfo>,
    float_endian: FloatingPointEndian,
    hierarchy: Option<(HierarchyCompression, u64)>,
}

impl<R: Read + Seek> HeaderReader<R> {
    fn new(input: R) -> Self {
        HeaderReader {
            input,
            header: None,
            signals: None,
            data_sections: Vec::default(),
            float_endian: FloatingPointEndian::Little,
            hierarchy: None,
        }
    }

    fn header_incomplete(&self) -> bool {
        match &self.header {
            None => true,
            Some(h) => h.start_time == 0 && h.end_time == 0,
        }
    }

    fn read_header(&mut self) -> Result<()> {
        let section_length = read_u64(&mut self.input)?;
        assert_eq!(section_length, 329);
        let start_time = read_u64(&mut self.input)?;
        let end_time = read_u64(&mut self.input)?;
        let float_endian = determine_f64_endian(&mut self.input, DOUBLE_ENDIAN_TEST)?;
        self.float_endian = float_endian;
        let memory_used_by_writer = read_u64(&mut self.input)?;
        let scope_count = read_u64(&mut self.input)?;
        let var_count = read_u64(&mut self.input)?;
        let max_var_id_code = read_u64(&mut self.input)?;
        let vc_section_count = read_u64(&mut self.input)?;
        let timescale_exponent = read_i8(&mut self.input)?;
        let version = read_c_str_fixed_length(&mut self.input, 128)?;
        // this size was reduced compared to what is documented in block_format.txt
        let date = read_c_str_fixed_length(&mut self.input, 119)?;
        let file_type = FileType::try_from(read_u8(&mut self.input)?)?;
        let time_zero = read_u64(&mut self.input)?;

        let header = Header {
            start_time,
            end_time,
            memory_used_by_writer,
            scope_count,
            var_count,
            max_var_id_code,
            vc_section_count,
            timescale_exponent,
            version,
            date,
            file_type,
            time_zero,
        };
        self.header = Some(header);
        Ok(())
    }

    fn read_data(&mut self, tpe: &BlockType) -> Result<()> {
        let file_offset = self.input.stream_position()?;
        // this is the data section
        let section_length = read_u64(&mut self.input)?;
        let start_time = read_u64(&mut self.input)?;
        let end_time = read_u64(&mut self.input)?;
        self.skip(section_length, 3 * 8)?;
        let kind = DataSectionKind::from_block_type(tpe).unwrap();
        let info = DataSectionInfo {
            file_offset,
            start_time,
            end_time,
            kind,
        };
        self.data_sections.push(info);
        Ok(())
    }

    fn skip(&mut self, section_length: u64, already_read: i64) -> Result<u64> {
        Ok(self
            .input
            .seek(SeekFrom::Current((section_length as i64) - already_read))?)
    }

    fn read_geometry(&mut self) -> Result<()> {
        let section_length = read_u64(&mut self.input)?;
        // skip this section if the header is not complete
        if self.header_incomplete() {
            self.skip(section_length, 8)?;
            return Ok(());
        }

        let uncompressed_length = read_u64(&mut self.input)?;
        let max_handle = read_u64(&mut self.input)?;
        let compressed_length = section_length - 3 * 8;

        let bytes = read_compressed_bytes(&mut self.input, uncompressed_length, compressed_length)?;

        let mut longest_signal_value_len = 32;
        let mut lengths: Vec<u32> = Vec::with_capacity(max_handle as usize);
        let mut types: Vec<FstVarType> = Vec::with_capacity(max_handle as usize);
        let mut byte_reader: &[u8] = &bytes;

        for _ii in 0..max_handle {
            let (value, _) = read_variant_u32(&mut byte_reader)?;
            let (length, tpe) = if value == 0 {
                (8, FstVarType::Real)
            } else {
                let length = if value != 0xFFFFFFFFu32 { value } else { 0 };
                if length > longest_signal_value_len {
                    // TODO: is this code ever run?
                    longest_signal_value_len = length;
                }
                (length, FstVarType::Wire)
            };
            lengths.push(length);
            types.push(tpe);
        }
        self.signals = Some(Signals { lengths, types });

        Ok(())
    }

    fn read_hierarchy(&mut self, compression: HierarchyCompression) -> Result<()> {
        let file_offset = self.input.stream_position()?;
        // this is the data section
        let section_length = read_u64(&mut self.input)?;
        self.skip(section_length, 8)?;
        assert!(
            self.hierarchy.is_none(),
            "Only a single hierarchy block is expected!"
        );
        self.hierarchy = Some((compression, file_offset));
        Ok(())
    }

    fn read(&mut self) -> Result<()> {
        loop {
            let block_tpe = match read_block_tpe(&mut self.input) {
                Err(ReaderError {
                    kind: ReaderErrorKind::IO(e),
                }) => {
                    //println!("{e:?}");
                    break;
                }
                Err(other) => return Err(other),
                Ok(tpe) => tpe,
            };
            // println!("{block_tpe:?}");
            match block_tpe {
                BlockType::Header => self.read_header()?,
                BlockType::VcData => self.read_data(&block_tpe)?,
                BlockType::VcDataDynamicAlias => self.read_data(&block_tpe)?,
                BlockType::VcDataDynamicAlias2 => self.read_data(&block_tpe)?,
                BlockType::Blackout => todo!("blackout"),
                BlockType::Geometry => self.read_geometry()?,
                BlockType::Hierarchy => self.read_hierarchy(HierarchyCompression::ZLib)?,
                BlockType::HierarchyLZ4 => self.read_hierarchy(HierarchyCompression::Lz4)?,
                BlockType::HierarchyLZ4Duo => self.read_hierarchy(HierarchyCompression::Lz4Duo)?,
                BlockType::GZipWrapper => todo!(),
                BlockType::Skip => todo!(),
            };
        }
        Ok(())
    }

    fn into_input_and_meta_data(mut self) -> Result<(R, MetaData)> {
        self.input.seek(SeekFrom::Start(0))?;
        let meta = MetaData {
            header: self.header.unwrap(),
            signals: self.signals.unwrap(),
            data_sections: self.data_sections,
            float_endian: self.float_endian,
            hierarchy_compression: self.hierarchy.unwrap().0,
            hierarchy_offset: self.hierarchy.unwrap().1,
        };
        Ok((self.input, meta))
    }
}

fn read_hierarchy_bytes(
    input: &mut (impl Read + Seek),
    compression: HierarchyCompression,
) -> Result<Vec<u8>> {
    let section_length = read_u64(input)? as usize;
    let uncompressed_length = read_u64(input)? as usize;
    let compressed_length = section_length - 2 * 8;

    let bytes = match compression {
        HierarchyCompression::ZLib => {
            let start = input.stream_position().unwrap();
            let mut d = flate2::read::GzDecoder::new(input);
            let mut uncompressed: Vec<u8> = Vec::with_capacity(uncompressed_length as usize);
            d.read_to_end(&mut uncompressed)?;
            // the decoder often reads more bytes than it should, we fix this here
            d.into_inner()
                .seek(SeekFrom::Start(start + compressed_length as u64))?;
            uncompressed
        }
        HierarchyCompression::Lz4 => {
            let compressed = read_bytes(input, compressed_length)?;
            lz4_flex::decompress(&compressed, uncompressed_length)?
        }
        HierarchyCompression::Lz4Duo => todo!("Implement LZ4 Duo!"),
    };
    assert_eq!(bytes.len(), uncompressed_length);
    Ok(bytes)
}

fn parse_misc_attribute(
    name: String,
    tpe: MiscType,
    arg: u64,
    arg2: Option<u64>,
) -> FstHierarchyEntry {
    match tpe {
        MiscType::Comment => todo!("comment Attribute"),
        MiscType::EnvVar => todo!("EnvVar Attribute"),
        MiscType::SupVar => todo!("SupVar Attribute"),
        MiscType::PathName => FstHierarchyEntry::PathName { name, id: arg },
        MiscType::SourceStem => FstHierarchyEntry::SourceStem {
            is_instantiation: false,
            path_id: arg2.unwrap(),
            line: arg,
        },
        MiscType::SourceInstantiationStem => FstHierarchyEntry::SourceStem {
            is_instantiation: true,
            path_id: arg2.unwrap(),
            line: arg,
        },
        MiscType::ValueList => todo!("ValueList Attribute"),
        MiscType::EnumTable => todo!("EnumTable Attribute"),
        MiscType::Unknown => todo!("unknown Attribute"),
    }
}

fn read_hierarchy_entry(
    input: &mut impl Read,
    handle_count: &mut u32,
) -> Result<Option<FstHierarchyEntry>> {
    let entry_tpe = match read_u8(input) {
        Ok(tpe) => tpe,
        Err(_) => return Ok(None),
    };
    let entry = match entry_tpe {
        254 => {
            // VcdScope (ScopeType)
            let tpe = FstScopeType::try_from_primitive(read_u8(input)?)?;
            let name = read_c_str(input, HIERARCHY_NAME_MAX_SIZE)?;
            let component = read_c_str(input, HIERARCHY_NAME_MAX_SIZE)?;
            FstHierarchyEntry::Scope {
                tpe,
                name,
                component,
            }
        }
        0..=29 => {
            // VcdEvent ... SvShortReal (VariableType)
            let tpe = FstVarType::try_from_primitive(entry_tpe)?;
            let direction = FstVarDirection::try_from_primitive(read_u8(input)?)?;
            let name = read_c_str(input, HIERARCHY_NAME_MAX_SIZE)?;
            let (raw_length, _) = read_variant_u32(input)?;
            let length = if tpe == FstVarType::Port {
                // remove delimiting spaces and adjust signal size
                (raw_length - 2) / 3
            } else {
                raw_length
            };
            let (alias, _) = read_variant_u32(input)?;
            let (is_alias, handle) = if alias == 0 {
                *handle_count += 1;
                (false, FstSignalHandle::new(*handle_count))
            } else {
                (true, FstSignalHandle::new(alias))
            };
            FstHierarchyEntry::Var {
                tpe,
                direction,
                name,
                length,
                handle,
                is_alias,
            }
        }
        255 => {
            // VcdUpScope (ScopeType)
            FstHierarchyEntry::UpScope
        }
        252 => {
            let tpe = AttributeType::try_from_primitive(read_u8(input)?)?;
            let subtype = MiscType::try_from_primitive(read_u8(input)?)?;
            let name = read_c_str(input, HIERARCHY_ATTRIBUTE_MAX_SIZE)?;
            let arg = read_variant_u64(input)?;
            match tpe {
                AttributeType::Misc => {
                    let arg2 = if subtype == MiscType::SourceStem
                        || subtype == MiscType::SourceInstantiationStem
                    {
                        Some(read_variant_u64(&mut name.as_bytes())?)
                    } else {
                        None
                    };
                    parse_misc_attribute(name, subtype, arg, arg2)
                }
                AttributeType::Array => todo!("ARRAY attributes"),
                AttributeType::Enum => todo!("ENUM attributes"),
                AttributeType::Pack => todo!("PACK attributes"),
            }
        }
        253 => {
            // GenAttributeEnd (ScopeType)
            FstHierarchyEntry::AttributeEnd
        }

        other => todo!("Deal with hierarchy entry of type: {other}"),
    };

    Ok(Some(entry))
}

type BitMask = BitVec<u8, Msb0>;

struct DataFilter {
    start: u64,
    end: u64,
    signals: BitMask,
}

struct DataReader<'a, R: Read + Seek, F: FnMut(u64, FstSignalHandle, &str)> {
    input: &'a mut R,
    meta: &'a MetaData,
    filter: &'a DataFilter,
    callback: &'a mut F,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DataSectionKind {
    Standard,
    DynamicAlias,
    DynamicAlias2,
}

#[derive(Debug, Clone, Copy)]
enum ValueChangePackType {
    Lz4,
    FastLz,
    Zlib,
}

impl ValueChangePackType {
    fn from_u8(value: u8) -> Self {
        match value {
            b'4' => ValueChangePackType::Lz4,
            b'F' => ValueChangePackType::FastLz,
            _ => ValueChangePackType::Zlib,
        }
    }
}

impl DataSectionKind {
    fn from_block_type(tpe: &BlockType) -> Option<Self> {
        match tpe {
            BlockType::VcData => Some(DataSectionKind::Standard),
            BlockType::VcDataDynamicAlias => Some(DataSectionKind::DynamicAlias),
            BlockType::VcDataDynamicAlias2 => Some(DataSectionKind::DynamicAlias2),
            _ => None,
        }
    }
}

const RCV_STR: [u8; 8] = [b'x', b'z', b'h', b'u', b'w', b'l', b'-', b'?'];
#[inline]
fn one_bit_signal_value_to_char(vli: u32) -> u8 {
    if (vli & 1) == 0 {
        (((vli >> 1) & 1) as u8) | b'0'
    } else {
        RCV_STR[((vli >> 1) & 7) as usize]
    }
}

/// Decodes a digital (1/0) signal. This is indicated by bit0 in vli being cleared.
#[inline]
fn multi_bit_digital_signal_to_chars(bytes: &[u8], len: usize) -> Vec<u8> {
    let mut chars = Vec::with_capacity(len);
    for ii in 0..len {
        let byte_id = ii / 8;
        let bit_id = 7 - (ii & 7);
        let bit = (bytes[byte_id] >> bit_id) & 1;
        chars.push(bit | b'0');
    }
    chars
}

#[inline]
fn int_div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn read_one_bit_signal_time_delta(bytes: &[u8], offset: u32) -> Result<usize> {
    let mut slice = &bytes[(offset as usize)..];
    let (vli, _) = read_variant_u32(&mut slice)?;
    let shift_count = 2u32 << (vli & 1);
    Ok((vli >> shift_count) as usize)
}

fn read_multi_bit_signal_time_delta(bytes: &[u8], offset: u32) -> Result<usize> {
    let mut slice = &bytes[(offset as usize)..];
    let (vli, _) = read_variant_u32(&mut slice)?;
    Ok((vli >> 1) as usize)
}

impl<'a, R: Read + Seek, F: FnMut(u64, FstSignalHandle, &str)> DataReader<'a, R, F> {
    fn read_time_block(
        &mut self,
        section_start: u64,
        section_length: u64,
    ) -> Result<(u64, Vec<u64>)> {
        // the time block meta data is in the last 24 bytes at the end of the section
        self.input
            .seek(SeekFrom::Start(section_start + section_length - 3 * 8))?;
        let uncompressed_length = read_u64(self.input)?;
        let compressed_length = read_u64(&mut self.input)?;
        let number_of_items = read_u64(&mut self.input)?;
        assert!(compressed_length <= section_length);

        // now that we know how long the block actually is, we can go back to it
        self.input
            .seek(SeekFrom::Current(-(3 * 8) - (compressed_length as i64)))?;
        let bytes = read_compressed_bytes(&mut self.input, uncompressed_length, compressed_length)?;
        let mut byte_reader: &[u8] = &bytes;
        let mut time_table: Vec<u64> = Vec::with_capacity(number_of_items as usize);
        let mut time_val: u64 = 0; // running time counter

        for _ in 0..number_of_items {
            let value = read_variant_u64(&mut byte_reader)?;
            time_val += value;
            time_table.push(time_val);
        }

        let time_section_length = compressed_length + 3 * 8;
        Ok((time_section_length, time_table))
    }

    fn read_frame(
        &mut self,
        section_start: u64,
        section_length: u64,
        start_time: u64,
    ) -> Result<()> {
        // we skip the section header (section_length, start_time, end_time, ???)
        self.input.seek(SeekFrom::Start(section_start + 4 * 8))?;
        let uncompressed_length = read_variant_u64(&mut self.input)?;
        let compressed_length = read_variant_u64(&mut self.input)?;
        let max_handle = read_variant_u64(&mut self.input)?;
        assert!(compressed_length <= section_length);
        let bytes = read_compressed_bytes(&mut self.input, uncompressed_length, compressed_length)?;

        let mut byte_reader: &[u8] = &bytes;
        for idx in 0..(max_handle as usize) {
            let signal_length = self.meta.signals.lengths[idx];
            if self.filter.signals[idx] {
                let signal_handle = FstSignalHandle::from_index(idx);
                // we do not support a "process_mask" for now, will read all signals
                match signal_length {
                    0 => {} // ignore since variable-length records have no initial value
                    len => {
                        let tpe = self.meta.signals.types[idx];
                        if tpe != FstVarType::Real {
                            let value = read_bytes(&mut byte_reader, len as usize)?;
                            (self.callback)(
                                start_time,
                                signal_handle,
                                std::str::from_utf8(&value)?,
                            );
                        } else {
                            let _value = read_f64(&mut byte_reader, self.meta.float_endian)?;
                            todo!("add support for reals")
                        }
                    }
                }
            } else {
                // skip
                self.input.seek(SeekFrom::Current(signal_length as i64))?;
            }
        }

        Ok(())
    }

    fn skip_frame(&mut self, section_start: u64) -> Result<()> {
        // we skip the section header (section_length, start_time, end_time, ???)
        self.input.seek(SeekFrom::Start(section_start + 4 * 8))?;
        let _uncompressed_length = read_variant_u64(&mut self.input)?;
        let compressed_length = read_variant_u64(&mut self.input)?;
        let _max_handle = read_variant_u64(&mut self.input)?;
        self.input
            .seek(SeekFrom::Current(compressed_length as i64))?;
        Ok(())
    }

    fn read_value_change_alias2(
        mut chain_bytes: &[u8],
        max_handle: u64,
    ) -> Result<(Vec<i64>, Vec<u32>, usize)> {
        let mut chain_table: Vec<i64> = Vec::with_capacity(max_handle as usize);
        let mut chain_table_lengths: Vec<u32> = vec![0u32; (max_handle + 1) as usize];
        let mut value = 0i64;
        let mut prev_alias = 0u32;
        let mut prev_idx = 0usize;
        while !chain_bytes.is_empty() {
            let idx = chain_table.len();
            let kind = chain_bytes[0];
            if (kind & 1) == 1 {
                let shval = read_variant_i64(&mut chain_bytes)? >> 1;
                if shval > 0 {
                    value += shval;
                    match chain_table.last() {
                        None => {} // this is the first iteration
                        Some(last_value) => {
                            let len = (value - last_value) as u32;
                            chain_table_lengths[prev_idx] = len;
                        }
                    };
                    prev_idx = idx;
                    chain_table.push(value);
                } else if shval < 0 {
                    chain_table.push(0);
                    prev_alias = shval as u32;
                    chain_table_lengths[idx] = prev_alias;
                } else {
                    chain_table.push(0);
                    chain_table_lengths[idx] = prev_alias;
                }
            } else {
                let (value, _) = read_variant_u32(&mut chain_bytes)?;
                let zeros = value >> 1;
                for _ in 0..zeros {
                    chain_table.push(0);
                }
            }
        }

        Ok((chain_table, chain_table_lengths, prev_idx))
    }

    fn read_value_change_alias(
        mut chain_bytes: &[u8],
        max_handle: u64,
    ) -> Result<(Vec<i64>, Vec<u32>, usize)> {
        let mut chain_table: Vec<i64> = Vec::with_capacity(max_handle as usize);
        let mut chain_table_lengths: Vec<u32> = vec![0u32; (max_handle + 1) as usize];
        let mut prev_idx = 0usize;
        let mut value = 0i64;
        while !chain_bytes.is_empty() {
            let (raw_val, _) = read_variant_u32(&mut chain_bytes)?;
            let idx = chain_table.len();
            if raw_val == 0 {
                chain_table.push(0);
                let (len, _) = read_variant_u32(&mut chain_bytes)?;
                chain_table_lengths[idx] = (-(len as i64)) as u32;
            } else if (raw_val & 1) == 1 {
                value += (raw_val as i64) >> 1;
                match chain_table.last() {
                    None => {} // this is the first iteration
                    Some(last_value) => {
                        let len = (value - last_value) as u32;
                        chain_table_lengths[prev_idx] = len;
                    }
                };
                chain_table.push(value);
                prev_idx = idx;
            } else {
                let zeros = raw_val >> 1;
                for _ in 0..zeros {
                    chain_table.push(0);
                }
            }
        }

        Ok((chain_table, chain_table_lengths, prev_idx))
    }

    fn fixup_chain_table(chain_table: &mut Vec<i64>, chain_lengths: &mut Vec<u32>) {
        assert_eq!(chain_table.len(), chain_lengths.len());
        for ii in 0..chain_table.len() {
            let v32 = chain_lengths[ii] as i32;
            if (v32 < 0) && (chain_table[ii] == 0) {
                // two's complement
                let v32_index = (-v32 - 1) as usize;
                if v32_index < ii {
                    // "sanity check"
                    chain_table[ii] = chain_table[v32_index];
                    chain_lengths[ii] = chain_lengths[v32_index];
                }
            }
        }
    }

    fn read_chain_table(
        &mut self,
        chain_len_offset: u64,
        section_kind: DataSectionKind,
        max_handle: u64,
        start: u64,
    ) -> Result<(Vec<i64>, Vec<u32>)> {
        self.input.seek(SeekFrom::Start(chain_len_offset))?;
        let chain_compressed_length = read_u64(&mut self.input)?;

        // the chain starts _chain_length_ bytes before the chain length
        let chain_start = chain_len_offset - chain_compressed_length;
        self.input.seek(SeekFrom::Start(chain_start))?;
        let chain_bytes = read_bytes(&mut self.input, chain_compressed_length as usize)?;

        let (mut chain_table, mut chain_table_lengths, prev_idx) =
            if section_kind == DataSectionKind::DynamicAlias2 {
                Self::read_value_change_alias2(&chain_bytes, max_handle)?
            } else {
                Self::read_value_change_alias(&chain_bytes, max_handle)?
            };
        let last_table_entry = (chain_start as i64) - (start as i64); // indx_pos - vc_start
        chain_table.push(last_table_entry);
        chain_table_lengths[prev_idx] = (last_table_entry - chain_table[prev_idx]) as u32;

        Self::fixup_chain_table(&mut chain_table, &mut chain_table_lengths);

        Ok((chain_table, chain_table_lengths))
    }

    fn read_packed_values(
        &mut self,
        chain_table_length: u32,
        packtpe: ValueChangePackType,
    ) -> Result<Vec<u8>> {
        let (value, skiplen) = read_variant_u32(&mut self.input)?;
        if value != 0 {
            let uncompressed_length = value as u64;
            let compressed_length = (chain_table_length - skiplen) as u64;
            // let compressed = read_bytes(&mut self.input, source_length as usize)?;
            let uncompressed: Vec<u8> = match packtpe {
                ValueChangePackType::Lz4 => todo!("Lz4"),
                ValueChangePackType::FastLz => todo!("FastLZ"),
                ValueChangePackType::Zlib => {
                    read_compressed_bytes(&mut self.input, uncompressed_length, compressed_length)?
                }
            };
            Ok(uncompressed)
        } else {
            let dest_length = chain_table_length - skiplen;
            let mut bytes = read_bytes(&mut self.input, dest_length as usize)?;
            Ok(bytes)
        }
    }

    fn read_value_changes(
        &mut self,
        section_kind: DataSectionKind,
        section_start: u64,
        section_length: u64,
        time_section_length: u64,
        time_table: &[u64],
    ) -> Result<()> {
        let max_handle = read_variant_u64(&mut self.input)?;
        let vc_start = self.input.stream_position()?;
        let packtpe = ValueChangePackType::from_u8(read_u8(&mut self.input)?);

        // the chain length is right in front of the time section
        let chain_len_offset = section_start + section_length - time_section_length - 8;
        let (chain_table, chain_table_lengths) =
            self.read_chain_table(chain_len_offset, section_kind, max_handle, vc_start)?;

        // read data and create a bunch of pointers
        let mut mu: Vec<u8> = Vec::new();
        let mut head_pointer: Vec<u32> = Vec::with_capacity(max_handle as usize);
        let mut length_remaining: Vec<u32> = Vec::with_capacity(max_handle as usize);
        let mut scatter_pointer = vec![0u32; max_handle as usize];
        let mut tc_head = vec![0u32; std::cmp::max(1, time_table.len())];

        for (signal_idx, (entry, length)) in chain_table
            .iter()
            .zip(chain_table_lengths.iter())
            .take(max_handle as usize)
            .enumerate()
        {
            if *entry != 0 {
                if self.filter.signals[signal_idx] {
                    self.input
                        .seek(SeekFrom::Start((vc_start as i64 + entry) as u64))?;

                    let mut bytes = self.read_packed_values(*length, packtpe)?;
                    head_pointer.push(mu.len() as u32);
                    length_remaining.push(bytes.len() as u32);
                    mu.append(&mut bytes);

                    let tdelta = if self.meta.signals.lengths[signal_idx] == 1 {
                        read_one_bit_signal_time_delta(&mu, head_pointer[signal_idx])?
                    } else {
                        read_multi_bit_signal_time_delta(&mu, head_pointer[signal_idx])?
                    };
                    scatter_pointer[signal_idx] = tc_head[tdelta];
                    tc_head[tdelta] = signal_idx as u32 + 1;
                } else {
                    // add dummy values
                    head_pointer.push(1234);
                    length_remaining.push(1234);
                }
            }
        }

        for (time_id, time) in time_table.iter().enumerate() {
            // while we cannot ignore signal changes before the start of the window
            // (since the signal might retain values for multiple cycles),
            // signal changes after our window are completely useless
            if *time > self.filter.end {
                break;
            }
            // handles cannot be zero
            while tc_head[time_id] != 0 {
                let signal_id = (tc_head[time_id] - 1) as usize;
                let mut mu_slice = &mu.as_slice()[head_pointer[signal_id] as usize..];
                let (vli, skiplen) = read_variant_u32(&mut mu_slice)?;
                let signal_len = self.meta.signals.lengths[signal_id];
                let signal_handle = FstSignalHandle::from_index(signal_id);
                let len = match signal_len {
                    1 => {
                        let value = one_bit_signal_value_to_char(vli);
                        let value_buf = [value];
                        (self.callback)(*time, signal_handle, std::str::from_utf8(&value_buf)?);
                        0 // no additional bytes consumed
                    }
                    0 => {
                        let (len, skiplen2) = read_variant_u32(&mut mu_slice)?;
                        todo!("variable length signal support! {len} {skiplen2}")
                    }
                    len => {
                        let tpe = self.meta.signals.types[signal_id];
                        let signal_len = len as usize;
                        if tpe != FstVarType::Real {
                            let (value, len) = if (vli & 1) == 0 {
                                // if bit0 is zero -> 2-state
                                let read_len = int_div_ceil(signal_len, 8);
                                let bytes = read_bytes(&mut mu_slice, read_len)?;
                                (
                                    multi_bit_digital_signal_to_chars(&bytes, signal_len),
                                    read_len as u32,
                                )
                            } else {
                                (read_bytes(&mut mu_slice, signal_len)?, len)
                            };
                            (self.callback)(*time, signal_handle, std::str::from_utf8(&value)?);
                            len
                        } else {
                            todo!("implement support for reals")
                        }
                    }
                };

                // update pointers
                let total_skiplen = skiplen + len;
                head_pointer[signal_id] += total_skiplen;
                length_remaining[signal_id] -= total_skiplen;
                tc_head[time_id] = scatter_pointer[signal_id];
                scatter_pointer[signal_id] = 0;

                if length_remaining[signal_id] > 0 {
                    let tdelta = if signal_len == 1 {
                        read_one_bit_signal_time_delta(&mu, head_pointer[signal_id])?
                    } else {
                        read_multi_bit_signal_time_delta(&mu, head_pointer[signal_id])?
                    };
                    scatter_pointer[signal_id] = tc_head[time_id + tdelta];
                    tc_head[time_id + tdelta] = (signal_id + 1) as u32;
                }
            }
        }

        Ok(())
    }

    fn read(&mut self) -> Result<()> {
        let sections = self.meta.data_sections.clone();
        // filter out any sections which are not in our time window
        let relevant_sections = sections
            .iter()
            .filter(|s| self.filter.end >= s.start_time && s.end_time >= self.filter.start);
        for (sec_num, section) in relevant_sections.enumerate() {
            // skip to section
            self.input.seek(SeekFrom::Start(section.file_offset))?;
            let section_length = read_u64(&mut self.input)?;

            // verify meta-data
            let start_time = read_u64(&mut self.input)?;
            let end_time = read_u64(&mut self.input)?;
            assert_eq!(start_time, section.start_time);
            assert_eq!(end_time, section.end_time);
            let is_first_section = sec_num == 0;

            // 66 is for the potential fastlz overhead
            // let mem_required_for_traversal = read_u64(&mut self.input)? + 66;

            let (time_section_length, time_table) =
                self.read_time_block(section.file_offset, section_length)?;

            // only read frame if this is the first section and there is no other data for
            // the start time
            if is_first_section && time_table[0] > start_time {
                self.read_frame(section.file_offset, section_length, start_time)?;
            } else {
                self.skip_frame(section.file_offset)?;
            }

            self.read_value_changes(
                section.kind,
                section.file_offset,
                section_length,
                time_section_length,
                &time_table,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_variant_i64() {
        // a positive value from a real fst file (solution from gtkwave)
        let in1 = [0x13];
        assert_eq!(read_variant_i64(&mut in1.as_slice()).unwrap(), 19);
        // a negative value from a real fst file (solution from gtkwave)
        let in0 = [0x7b];
        assert_eq!(read_variant_i64(&mut in0.as_slice()).unwrap(), -5);
    }

    #[test]
    fn test_read_c_str_fixed_length() {
        let input = [b'h', b'i', 0u8, b'x'];
        assert_eq!(
            read_c_str_fixed_length(&mut input.as_slice(), 4).unwrap(),
            "hi"
        );
        let input2 = [b'h', b'i', b'i', 0u8, b'x'];
        assert_eq!(
            read_c_str_fixed_length(&mut input2.as_slice(), 5).unwrap(),
            "hii"
        );
    }
}
