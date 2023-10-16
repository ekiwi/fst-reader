// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
// Contains basic read and write operations for FST files.

use crate::types::*;
use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::io::{Read, Seek, SeekFrom, Write};

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
    /// Failed to parse the string contained in an enum table attribute.
    EnumTableString(),
    ParseInt(std::num::ParseIntError),
}
#[derive(Debug)]
pub struct ReaderError {
    pub(crate) kind: ReaderErrorKind,
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

impl From<std::num::ParseIntError> for ReaderError {
    fn from(value: std::num::ParseIntError) -> Self {
        let kind = ReaderErrorKind::ParseInt(value);
        ReaderError { kind }
    }
}

pub type ReadResult<T> = std::result::Result<T, ReaderError>;

pub type WriteResult<T> = std::result::Result<T, ReaderError>;

//////////////// Primitives

#[inline]
pub(crate) fn read_variant_u32(input: &mut impl Read) -> ReadResult<(u32, u32)> {
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
pub(crate) fn read_variant_i64(input: &mut impl Read) -> ReadResult<i64> {
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
pub(crate) fn read_variant_u64(input: &mut impl Read) -> ReadResult<u64> {
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
pub(crate) fn write_variant_u64(output: &mut impl Write, mut value: u64) -> WriteResult<()> {
    // often, the value is small
    if value <= 0x7f {
        let byte = [value as u8; 1];
        output.write_all(&byte)?;
        return Ok(());
    }

    let mut bytes = Vec::with_capacity(10);
    while value != 0 {
        let next_value = value >> 7;
        let mask: u8 = if next_value == 0 { 0 } else { 0x80 };
        bytes.push((value & 0x7f) as u8 | mask);
        value = next_value;
    }
    assert!(bytes.len() <= 10);
    output.write_all(&bytes)?;
    Ok(())
}

#[inline]
pub(crate) fn read_u64(input: &mut impl Read) -> ReadResult<u64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

#[inline]
pub(crate) fn write_u64(output: &mut impl Write, value: u64) -> WriteResult<()> {
    let buf = value.to_be_bytes();
    output.write_all(&buf)?;
    Ok(())
}

#[inline]
pub(crate) fn read_u8(input: &mut impl Read) -> ReadResult<u8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn write_u8(output: &mut impl Write, value: u8) -> WriteResult<()> {
    let buf = value.to_be_bytes();
    output.write_all(&buf)?;
    Ok(())
}

#[inline]
pub(crate) fn read_i8(input: &mut impl Read) -> ReadResult<i8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(i8::from_be_bytes(buf))
}

#[inline]
fn write_i8(output: &mut impl Write, value: i8) -> WriteResult<()> {
    let buf = value.to_be_bytes();
    output.write_all(&buf)?;
    Ok(())
}

pub(crate) fn read_c_str(input: &mut impl Read, max_len: usize) -> ReadResult<String> {
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
pub(crate) fn read_c_str_fixed_length(input: &mut impl Read, len: usize) -> ReadResult<String> {
    let mut bytes = read_bytes(input, len)?;
    let zero_index = bytes.iter().position(|b| *b == 0u8).unwrap_or(len - 1);
    let str_len = zero_index;
    bytes.truncate(str_len);
    Ok(String::from_utf8(bytes)?)
}

#[inline]
fn write_c_str_fixed_length(
    output: &mut impl Write,
    value: &str,
    max_len: usize,
) -> WriteResult<()> {
    let bytes = value.as_bytes();
    if bytes.len() >= max_len {
        todo!("Return error.")
    }
    output.write_all(&bytes)?;
    let zeros = vec![0u8; max_len - bytes.len()];
    output.write_all(&zeros)?;
    Ok(())
}

const RCV_STR: [u8; 8] = [b'x', b'z', b'h', b'u', b'w', b'l', b'-', b'?'];
#[inline]
pub(crate) fn one_bit_signal_value_to_char(vli: u32) -> u8 {
    if (vli & 1) == 0 {
        (((vli >> 1) & 1) as u8) | b'0'
    } else {
        RCV_STR[((vli >> 1) & 7) as usize]
    }
}

/// Decodes a digital (1/0) signal. This is indicated by bit0 in vli being cleared.
#[inline]
pub(crate) fn multi_bit_digital_signal_to_chars(bytes: &[u8], len: usize) -> Vec<u8> {
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
pub(crate) fn int_div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub(crate) fn read_one_bit_signal_time_delta(bytes: &[u8], offset: u32) -> ReadResult<usize> {
    let mut slice = &bytes[(offset as usize)..];
    let (vli, _) = read_variant_u32(&mut slice)?;
    let shift_count = 2u32 << (vli & 1);
    Ok((vli >> shift_count) as usize)
}

pub(crate) fn read_multi_bit_signal_time_delta(bytes: &[u8], offset: u32) -> ReadResult<usize> {
    let mut slice = &bytes[(offset as usize)..];
    let (vli, _) = read_variant_u32(&mut slice)?;
    Ok((vli >> 1) as usize)
}

/// Reads ZLib compressed bytes.
pub(crate) fn read_zlib_compressed_bytes(
    input: &mut (impl Read + Seek),
    uncompressed_length: u64,
    compressed_length: u64,
    allow_uncompressed: bool,
) -> ReadResult<Vec<u8>> {
    let bytes = if uncompressed_length == compressed_length && allow_uncompressed {
        read_bytes(input, compressed_length as usize)?
    } else {
        let start = input.stream_position().unwrap();
        let mut d = flate2::read::ZlibDecoder::new(input);
        let mut uncompressed: Vec<u8> = Vec::with_capacity(uncompressed_length as usize);
        d.read_to_end(&mut uncompressed)?;
        // sanity checks
        assert_eq!(d.total_out(), uncompressed_length);
        // the decoder often reads more bytes than it should, we fix this here
        d.into_inner()
            .seek(SeekFrom::Start(start + compressed_length))?;
        uncompressed
    };
    assert_eq!(bytes.len(), uncompressed_length as usize);
    Ok(bytes)
}

/// ZLib compresses bytes. If allow_uncompressed is true, we overwrite the compressed with the
/// uncompressed bytes if it turns out that the compressed bytes are longer.
pub(crate) fn write_compressed_bytes(
    output: &mut (impl Write + Seek),
    bytes: &[u8],
    compression_level: flate2::Compression,
    allow_uncompressed: bool,
) -> WriteResult<usize> {
    let start = output.stream_position()?;
    let mut d = flate2::write::ZlibEncoder::new(output, compression_level);
    d.write_all(bytes)?;
    d.flush()?;
    assert_eq!(d.total_in() as usize, bytes.len());
    let compressed_written = d.total_out() as usize;
    let output2 = d.finish()?;
    if !allow_uncompressed || compressed_written < bytes.len() {
        Ok(compressed_written)
    } else {
        // it turns out that the compression was futile!
        output2.seek(SeekFrom::Start(start))?;
        output2.write_all(bytes)?;
        Ok(bytes.len())
    }
}

#[inline]
pub(crate) fn read_bytes(input: &mut impl Read, len: usize) -> ReadResult<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    input.take(len as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

pub(crate) fn read_block_tpe(input: &mut impl Read) -> ReadResult<BlockType> {
    Ok(BlockType::try_from(read_u8(input)?)?)
}

pub(crate) fn determine_f64_endian(
    input: &mut impl Read,
    needle: f64,
) -> ReadResult<FloatingPointEndian> {
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

#[inline]
pub(crate) fn read_f64(input: &mut impl Read, endian: FloatingPointEndian) -> ReadResult<f64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    match endian {
        FloatingPointEndian::Little => Ok(f64::from_le_bytes(buf)),
        FloatingPointEndian::Big => Ok(f64::from_be_bytes(buf)),
    }
}

#[inline]
fn write_f64(output: &mut impl Write, value: f64) -> WriteResult<()> {
    // for f64, we have the option to use either LE or BE, we just need to be consistent
    let buf = value.to_le_bytes();
    output.write_all(&buf)?;
    Ok(())
}

fn read_lz4_compressed_bytes(
    input: &mut impl Read,
    uncompressed_length: usize,
    compressed_length: usize,
) -> ReadResult<Vec<u8>> {
    let compressed = read_bytes(input, compressed_length)?;
    let bytes = lz4_flex::decompress(&compressed, uncompressed_length)?;
    Ok(bytes)
}

//////////////// Header

const HEADER_LENGTH: u64 = 329;
const HEADER_VERSION_MAX_LEN: usize = 128;
const HEADER_DATE_MAX_LEN: usize = 119;
pub(crate) fn read_header(input: &mut impl Read) -> ReadResult<(Header, FloatingPointEndian)> {
    let section_length = read_u64(input)?;
    assert_eq!(section_length, HEADER_LENGTH);
    let start_time = read_u64(input)?;
    let end_time = read_u64(input)?;
    let float_endian = determine_f64_endian(input, DOUBLE_ENDIAN_TEST)?;
    let memory_used_by_writer = read_u64(input)?;
    let scope_count = read_u64(input)?;
    let var_count = read_u64(input)?;
    let max_var_id_code = read_u64(input)?;
    let vc_section_count = read_u64(input)?;
    let timescale_exponent = read_i8(input)?;
    let version = read_c_str_fixed_length(input, HEADER_VERSION_MAX_LEN)?;
    // this size was reduced compared to what is documented in block_format.txt
    let date = read_c_str_fixed_length(input, HEADER_DATE_MAX_LEN)?;
    let file_type = FileType::try_from(read_u8(input)?)?;
    let time_zero = read_u64(input)?;

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
    Ok((header, float_endian))
}

pub(crate) fn write_header(output: &mut impl Write, header: &Header) -> WriteResult<()> {
    write_u64(output, HEADER_LENGTH)?;
    write_u64(output, header.start_time)?;
    write_u64(output, header.end_time)?;
    write_f64(output, DOUBLE_ENDIAN_TEST)?;
    write_u64(output, header.memory_used_by_writer)?;
    write_u64(output, header.scope_count)?;
    write_u64(output, header.var_count)?;
    write_u64(output, header.max_var_id_code)?;
    write_u64(output, header.vc_section_count)?;
    write_i8(output, header.timescale_exponent)?;
    write_c_str_fixed_length(output, &header.version, HEADER_VERSION_MAX_LEN)?;
    write_c_str_fixed_length(output, &header.date, HEADER_DATE_MAX_LEN)?;
    write_u8(output, header.file_type as u8)?;
    write_u64(output, header.time_zero)?;
    Ok(())
}

//////////////// Geometry

pub(crate) fn read_geometry(input: &mut (impl Read + Seek)) -> ReadResult<Vec<SignalInfo>> {
    let section_length = read_u64(input)?;
    let uncompressed_length = read_u64(input)?;
    let max_handle = read_u64(input)?;
    let compressed_length = section_length - 3 * 8;

    let bytes = read_zlib_compressed_bytes(input, uncompressed_length, compressed_length, true)?;

    let mut signals: Vec<SignalInfo> = Vec::with_capacity(max_handle as usize);
    let mut byte_reader: &[u8] = &bytes;

    for _ii in 0..max_handle {
        let (value, _) = read_variant_u32(&mut byte_reader)?;
        signals.push(SignalInfo::from_file_format(value));
    }
    Ok(signals)
}

const GEOMETRY_COMPRESSION_LEVEL: flate2::Compression = flate2::Compression::best();

pub(crate) fn write_geometry(
    output: &mut (impl Write + Seek),
    signals: &Vec<SignalInfo>,
) -> WriteResult<()> {
    // remember start to fix the section length afterwards
    let start = output.stream_position()?;
    write_u64(output, 0)?; // dummy section length

    // write uncompressed signal info
    let mut bytes: Vec<u8> = Vec::with_capacity(signals.len() * 2);
    for signal in signals {
        write_variant_u64(&mut bytes, signal.to_file_format() as u64)?;
    }
    let uncompressed_length = bytes.len() as u64;
    write_u64(output, uncompressed_length)?;
    let max_handle = signals.len() as u64;
    write_u64(output, max_handle)?;

    // compress signals
    let compressed_len =
        write_compressed_bytes(output, &bytes, GEOMETRY_COMPRESSION_LEVEL, true)? as u64;

    // fix section length
    let section_length = compressed_len + 3 * 8;
    let end = output.stream_position()?;
    output.seek(SeekFrom::Start(start))?;
    write_u64(output, section_length)?;
    output.seek(SeekFrom::Start(end))?;

    Ok(())
}

//////////////// Blackout

pub(crate) fn read_blackout(input: &mut (impl Read + Seek)) -> ReadResult<Vec<BlackoutData>> {
    let (num_blackouts, _) = read_variant_u32(input)?;
    let mut blackouts = Vec::with_capacity(num_blackouts as usize);
    let mut current_blackout = 0u64;
    for ii in 0..num_blackouts {
        let activity = read_u8(input)? != 0;
        let delta = read_variant_u64(input)?;
        current_blackout += delta;
        let bo = BlackoutData {
            time: current_blackout,
            contains_activity: activity,
        };
        blackouts.push(bo);
    }
    Ok(blackouts)
}

//////////////// Hierarchy

pub(crate) fn read_hierarchy_bytes(
    input: &mut (impl Read + Seek),
    compression: HierarchyCompression,
) -> ReadResult<Vec<u8>> {
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
            read_lz4_compressed_bytes(input, uncompressed_length, compressed_length)?
        }
        HierarchyCompression::Lz4Duo => todo!("Implement LZ4 Duo!"),
    };
    assert_eq!(bytes.len(), uncompressed_length);
    Ok(bytes)
}

pub(crate) fn write_hierarchy(output: &mut (impl Write + Seek)) -> WriteResult<()> {
    todo!();
}

fn enum_table_from_string(value: String, handle: u64) -> ReadResult<FstHierarchyEntry> {
    let parts: Vec<&str> = value.split(" ").collect();
    if parts.len() < 2 {
        return Err(ReaderError {
            kind: ReaderErrorKind::EnumTableString(),
        });
    }
    let name = parts[0].to_string();
    let element_count = usize::from_str_radix(parts[1], 10)?;
    if parts.len() != 2 + element_count * 2 {
        return Err(ReaderError {
            kind: ReaderErrorKind::EnumTableString(),
        });
    }
    let mut mapping = Vec::with_capacity(element_count);
    for ii in 0..element_count {
        let name = parts[2 + ii].to_string();
        let value = parts[2 + element_count + ii].to_string();
        mapping.push((value, name));
    }
    // TODO: deal with correct de-escaping
    Ok(FstHierarchyEntry::EnumTable {
        name,
        handle,
        mapping,
    })
}

fn parse_misc_attribute(
    name: String,
    tpe: MiscType,
    arg: u64,
    arg2: Option<u64>,
) -> ReadResult<FstHierarchyEntry> {
    let res = match tpe {
        MiscType::Comment => FstHierarchyEntry::Comment { string: name },
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
        MiscType::EnumTable => {
            if name.is_empty() {
                FstHierarchyEntry::EnumTableRef { handle: arg }
            } else {
                enum_table_from_string(name, arg)?
            }
        }
        MiscType::Unknown => todo!("unknown Attribute"),
    };
    Ok(res)
}

pub(crate) fn read_hierarchy_entry(
    input: &mut impl Read,
    handle_count: &mut u32,
) -> ReadResult<Option<FstHierarchyEntry>> {
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
                    parse_misc_attribute(name, subtype, arg, arg2)?
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

//////////////// Vale Change Data

// for debugging
fn print_python_bytes(bytes: &[u8]) {
    print!("b\"");
    for bb in bytes {
        print!("\\x{:02x}", bb);
    }
    println!("\"");
}

pub(crate) fn read_packed_signal_values(
    input: &mut (impl Read + Seek),
    len: u32,
    tpe: ValueChangePackType,
) -> ReadResult<Vec<u8>> {
    let (value, skiplen) = read_variant_u32(input)?;
    if value != 0 {
        let uncompressed_length = value as u64;
        let uncompressed: Vec<u8> = match tpe {
            ValueChangePackType::Lz4 => {
                let compressed_length = (len - skiplen) as u64;
                read_lz4_compressed_bytes(
                    input,
                    uncompressed_length as usize,
                    compressed_length as usize,
                )?
            }
            ValueChangePackType::FastLz => todo!("FastLZ"),
            ValueChangePackType::Zlib => {
                let compressed_length = len as u64;
                // Important: for signals, we do not skip decompression,
                // even if the compressed and uncompressed length are the same
                read_zlib_compressed_bytes(input, uncompressed_length, compressed_length, false)?
            }
        };
        Ok(uncompressed)
    } else {
        let dest_length = len - skiplen;
        let bytes = read_bytes(input, dest_length as usize)?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_read_variant_i64() {
        // a positive value from a real fst file (solution from gtkwave)
        let in1 = [0x13];
        assert_eq!(read_variant_i64(&mut in1.as_slice()).unwrap(), 19);
        // a negative value from a real fst file (solution from gtkwave)
        let in0 = [0x7b];
        assert_eq!(read_variant_i64(&mut in0.as_slice()).unwrap(), -5);
    }

    proptest! {
         #[test]
        fn test_read_write_variant_u64(value: u64) {
            let mut buf = std::io::Cursor::new(vec![0u8; 24]);
            write_variant_u64(&mut buf, value).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            let read_value = read_variant_u64(&mut buf).unwrap();
            assert_eq!(read_value, value);
        }
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

    proptest! {
        #[test]
        fn test_write_c_str_fixed_length(string: String, max_len in 1 .. 400usize) {
            let string_bytes: &[u8] = string.as_bytes();
            prop_assume!(string_bytes.len() < max_len);
            prop_assume!(!string_bytes.contains(&0u8));
            let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
            write_c_str_fixed_length(&mut buf, &string, max_len).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            assert_eq!(
                read_c_str_fixed_length(&mut buf, max_len).unwrap(),
                string
            );
        }
    }

    proptest! {
        #[test]
        fn test_read_write_header(header: Header) {
            // return early if the header strings are too long
            prop_assume!(header.version.len() <= HEADER_VERSION_MAX_LEN);
            prop_assume!(header.date.len() <= HEADER_DATE_MAX_LEN );

            let mut buf = [0u8; 512];
            write_header(&mut buf.as_mut(), &header).unwrap();
            let (actual_header, endian) = read_header(&mut buf.as_slice()).unwrap();
            assert_eq!(endian, FloatingPointEndian::Little);
            assert_eq!(actual_header, header);
        }
    }

    proptest! {
        #[test]
        fn test_compress_bytes(bytes: Vec<u8>, allow_uncompressed: bool) {
            let mut buf = std::io::Cursor::new(vec![0u8; bytes.len() * 2]);
            let compressed_len = write_compressed_bytes(&mut buf, &bytes, flate2::Compression::new(9), allow_uncompressed).unwrap();
            if allow_uncompressed {
                assert!(compressed_len <= bytes.len());
            }
            buf.seek(SeekFrom::Start(0)).unwrap();
            let uncompressed = read_zlib_compressed_bytes(&mut buf, bytes.len() as u64, compressed_len as u64, allow_uncompressed).unwrap();
            assert_eq!(uncompressed, bytes);
        }
    }

    proptest! {
        #[test]
        fn test_read_write_geometry(signals: Vec<SignalInfo>) {
            let max_len = signals.len() * 4 + 3 * 8;
            let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
            write_geometry(&mut buf, &signals).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            let actual = read_geometry(&mut buf).unwrap();
            assert_eq!(actual.len(), signals.len());
            assert_eq!(actual, signals);
        }
    }
}
