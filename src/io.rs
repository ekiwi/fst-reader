// Copyright 2023 The Regents of the University of California
// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
// Contains basic read and write operations for FST files.

use crate::types::*;
use crate::FstSignalValue;
use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::cmp::Ordering;
#[cfg(test)]
use std::io::Write;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZeroU32;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("failed to read a null terminated string because it exceeds the expected size of {0} bytes.\n{1}")]
    CStringTooLong(usize, String),
    #[error("failed to parse an enum table string: {0}\n{1}")]
    EnumTableString(String, String),
    #[error("failed to read leb128 integer, more than the expected {0} bits")]
    Leb128(u32),
    #[error("failed to parse an integer")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("failed to decompress with lz4")]
    Lz4Decompress(#[from] lz4_flex::block::DecompressError),
    #[error("failed to decompress with zlib")]
    ZLibDecompress(#[from] miniz_oxide::inflate::DecompressError),
    #[error("failed to parse a gzip header: {0}")]
    GZipHeader(String),
    #[error("failed to decompress gzip stream: {0}")]
    GZipBody(String),
    #[error("failed to decode string")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("failed to decode string")]
    Utf8String(#[from] std::string::FromUtf8Error),
    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),
    #[error("The FST file is still being compressed into its final GZIP wrapper.")]
    NotFinishedCompressing(),
    #[error("Unexpected block type")]
    BlockType(#[from] TryFromPrimitiveError<BlockType>),
    #[error("Unexpected file type")]
    FileType(#[from] TryFromPrimitiveError<FileType>),
    #[error("Unexpected vhdl variable type")]
    FstVhdlVarType(#[from] TryFromPrimitiveError<FstVhdlVarType>),
    #[error("Unexpected vhdl data type")]
    FstVhdlDataType(#[from] TryFromPrimitiveError<FstVhdlDataType>),
    #[error("Unexpected variable type")]
    FstVarType(#[from] TryFromPrimitiveError<FstVarType>),
    #[error("Unexpected scope type")]
    FstScopeType(#[from] TryFromPrimitiveError<FstScopeType>),
    #[error("Unexpected variable direction")]
    FstVarDirection(#[from] TryFromPrimitiveError<FstVarDirection>),
    #[error("Unexpected attribute type")]
    AttributeType(#[from] TryFromPrimitiveError<AttributeType>),
    #[error("Unexpected misc attribute type")]
    MiscType(#[from] TryFromPrimitiveError<MiscType>),
    #[error("The FST file is incomplete: geometry block is missing.")]
    MissingGeometry(),
    #[error("The FST file is incomplete: hierarchy block is missing.")]
    MissingHierarchy(),
}

pub type ReadResult<T> = Result<T, ReaderError>;

#[cfg(test)]
pub type WriteResult<T> = Result<T, ReaderError>;

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
    Err(ReaderError::Leb128(32))
}

#[inline]
pub(crate) fn read_variant_i64(input: &mut impl Read) -> ReadResult<i64> {
    let mut byte = [0u8; 1];
    let mut res = 0u64;
    // 64bit / 7bit = ~9.1
    for ii in 0..10 {
        input.read_exact(&mut byte)?;
        let value = (byte[0] & 0x7f) as u64;
        let shift_by = 7 * ii;
        res |= value << shift_by;
        if (byte[0] & 0x80) == 0 {
            // sign extend
            let sign_bit_set = (byte[0] & 0x40) != 0;
            if (shift_by + 7) < u64::BITS && sign_bit_set {
                res |= u64::MAX << (shift_by + 7);
            }
            return Ok(res as i64);
        }
    }
    Err(ReaderError::Leb128(64))
}

#[inline]
pub(crate) fn read_variant_u64(input: &mut impl Read) -> ReadResult<(u64, usize)> {
    let mut byte = [0u8; 1];
    let mut res = 0u64;
    for ii in 0..10 {
        // 64bit / 7bit = ~9.1
        input.read_exact(&mut byte)?;
        let value = (byte[0] as u64) & 0x7f;
        res |= value << (7 * ii);
        if (byte[0] & 0x80) == 0 {
            return Ok((res, ii + 1));
        }
    }
    Err(ReaderError::Leb128(64))
}

#[cfg(test)]
#[inline]
pub(crate) fn write_variant_u64(output: &mut impl Write, mut value: u64) -> WriteResult<usize> {
    // often, the value is small
    if value <= 0x7f {
        let byte = [value as u8; 1];
        output.write_all(&byte)?;
        return Ok(1);
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
    Ok(bytes.len())
}

#[cfg(test)]
#[inline]
pub(crate) fn write_variant_i64(output: &mut impl Write, mut value: i64) -> WriteResult<usize> {
    // often, the value is small
    if value <= 63 && value >= -64 {
        let byte = [value as u8 & 0x7f; 1];
        output.write_all(&byte)?;
        return Ok(1);
    }

    // calculate the number of bits we need to represent
    let bits = if value >= 0 {
        64 - value.leading_zeros() + 1
    } else {
        64 - value.leading_ones() + 1
    };
    let num_bytes = bits.div_ceil(7) as usize;

    let mut bytes = Vec::with_capacity(num_bytes);
    for ii in 0..num_bytes {
        let mark = if ii == num_bytes - 1 { 0 } else { 0x80 };
        bytes.push((value & 0x7f) as u8 | mark);
        value >>= 7;
    }
    output.write_all(&bytes)?;
    Ok(bytes.len())
}

#[cfg(test)]
#[inline]
pub(crate) fn write_variant_u32(output: &mut impl Write, value: u32) -> WriteResult<usize> {
    write_variant_u64(output, value as u64)
}

#[inline]
pub(crate) fn read_u64(input: &mut impl Read) -> ReadResult<u64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
            return Ok(String::from_utf8(bytes)?);
        } else {
            bytes.push(byte);
        }
    }
    Err(ReaderError::CStringTooLong(
        max_len,
        String::from_utf8_lossy(&bytes).to_string(),
    ))
}

#[cfg(test)]
fn write_c_str(output: &mut impl Write, value: &str) -> WriteResult<()> {
    let bytes = value.as_bytes();
    output.write_all(bytes)?;
    write_u8(output, 0)?;
    Ok(())
}

#[inline] // inline to specialize on length
pub(crate) fn read_c_str_fixed_length(input: &mut impl Read, len: usize) -> ReadResult<String> {
    let mut bytes = read_bytes(input, len)?;
    let zero_index = bytes.iter().position(|b| *b == 0u8).unwrap_or(len - 1);
    let str_len = zero_index;
    bytes.truncate(str_len);
    Ok(String::from_utf8(bytes)?)
}

#[cfg(test)]
#[cfg(test)]
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
    output.write_all(bytes)?;
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
pub(crate) fn multi_bit_digital_signal_to_chars(bytes: &[u8], len: usize, output: &mut Vec<u8>) {
    output.resize(len, 0);
    for (ii, out) in output.iter_mut().enumerate() {
        let byte_id = ii / 8;
        let bit_id = 7 - (ii & 7);
        let bit = (bytes[byte_id] >> bit_id) & 1;
        *out = bit | b'0';
    }
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
        let start = input.stream_position()?;

        // read first byte to check which compression is used.
        let first_byte = read_u8(input)?;
        input.seek(SeekFrom::Start(start))?;
        // for zlib compression, the first byte should be 0x78
        let is_zlib = first_byte == 0x78;
        debug_assert!(is_zlib, "expected a zlib compressed block!");

        let compressed = read_bytes(input, compressed_length as usize)?;

        miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
            compressed.as_slice(),
            uncompressed_length as usize,
        )?
    };
    assert_eq!(bytes.len(), uncompressed_length as usize);
    Ok(bytes)
}

/// ZLib compresses bytes. If allow_uncompressed is true, we overwrite the compressed with the
/// uncompressed bytes if it turns out that the compressed bytes are longer.
#[cfg(test)]
pub(crate) fn write_compressed_bytes(
    output: &mut (impl Write + Seek),
    bytes: &[u8],
    compression_level: u8,
    allow_uncompressed: bool,
) -> WriteResult<usize> {
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(bytes, compression_level);
    if !allow_uncompressed || compressed.len() < bytes.len() {
        output.write_all(compressed.as_slice())?;
        Ok(compressed.len())
    } else {
        // it turns out that the compression was futile!
        output.write_all(bytes)?;
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn write_geometry(
    output: &mut (impl Write + Seek),
    signals: &Vec<SignalInfo>,
    compression: u8,
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
    let compressed_len = write_compressed_bytes(output, &bytes, compression, true)? as u64;

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
    // remember start for later sanity check
    let start = input.stream_position()?;
    let section_length = read_u64(input)?;
    let (num_blackouts, _) = read_variant_u32(input)?;
    let mut blackouts = Vec::with_capacity(num_blackouts as usize);
    let mut current_blackout = 0u64;
    for _ in 0..num_blackouts {
        let activity = read_u8(input)? != 0;
        let (delta, _) = read_variant_u64(input)?;
        current_blackout += delta;
        let bo = BlackoutData {
            time: current_blackout,
            contains_activity: activity,
        };
        blackouts.push(bo);
    }
    let end = input.stream_position()?;
    assert_eq!(start + section_length, end);
    Ok(blackouts)
}

#[cfg(test)]
pub(crate) fn write_blackout(
    output: &mut (impl Write + Seek),
    blackouts: &[BlackoutData],
) -> WriteResult<()> {
    // remember start to fix the section length afterwards
    let start = output.stream_position()?;
    write_u64(output, 0)?; // dummy section length

    let num_blackouts = blackouts.len() as u32;
    write_variant_u32(output, num_blackouts)?;

    let mut last_blackout = 0u64;
    for blackout in blackouts {
        let activity_byte = if blackout.contains_activity { 1 } else { 0 };
        write_u8(output, activity_byte)?;
        let delta = blackout.time - last_blackout;
        last_blackout = blackout.time;
        write_variant_u64(output, delta)?;
    }

    // fix section length
    let end = output.stream_position()?;
    output.seek(SeekFrom::Start(start))?;
    write_u64(output, end - start)?;
    output.seek(SeekFrom::Start(end))?;

    Ok(())
}

//////////////// Hierarchy
#[cfg(test)]
const HIERARCHY_GZIP_COMPRESSION_LEVEL: u8 = 4;

/// uncompresses zlib compressed bytes with a gzip header
fn read_gzip_compressed_bytes(
    input: &mut impl Read,
    uncompressed_len: usize,
    compressed_len: usize,
) -> ReadResult<Vec<u8>> {
    read_gzip_header(input)?;
    // we do not care about other header bytes
    let data = read_bytes(input, compressed_len - 10)?;
    let uncompressed =
        miniz_oxide::inflate::decompress_to_vec_with_limit(data.as_slice(), uncompressed_len)?;
    debug_assert_eq!(uncompressed.len(), uncompressed_len);
    Ok(uncompressed)
}

pub(crate) fn read_gzip_header(input: &mut impl Read) -> ReadResult<()> {
    let header = read_bytes(input, 10)?;
    let correct_magic = header[0] == 0x1f && header[1] == 0x8b;
    if !correct_magic {
        return Err(ReaderError::GZipHeader(format!(
            "expected magic bytes (0x1f, 0x8b) got {header:x?}"
        )));
    }
    let is_deflate_compressed = header[2] == 8;
    if !is_deflate_compressed {
        return Err(ReaderError::GZipHeader(format!(
            "expected deflate compression (8) got {:x?}",
            header[2]
        )));
    }
    let flag = header[3];
    if flag != 0 {
        return Err(ReaderError::GZipHeader(format!(
            "TODO currently extra flags are not supported {flag}"
        )));
    }
    Ok(())
}

pub(crate) fn read_hierarchy_bytes(
    input: &mut (impl Read + Seek),
    compression: HierarchyCompression,
) -> ReadResult<Vec<u8>> {
    let section_length = read_u64(input)? as usize;
    let uncompressed_length = read_u64(input)? as usize;
    let compressed_length = section_length - 2 * 8;

    let bytes = match compression {
        HierarchyCompression::ZLib => {
            read_gzip_compressed_bytes(input, uncompressed_length, compressed_length)?
        }
        HierarchyCompression::Lz4 => {
            read_lz4_compressed_bytes(input, uncompressed_length, compressed_length)?
        }
        HierarchyCompression::Lz4Duo => {
            // the length after the _first_ decompression
            let (len, skiplen) = read_variant_u64(input)?;
            let lvl1_len = len as usize;
            let lvl1 = read_lz4_compressed_bytes(input, lvl1_len, compressed_length - skiplen)?;
            let mut lvl1_reader = lvl1.as_slice();
            read_lz4_compressed_bytes(&mut lvl1_reader, uncompressed_length, lvl1_len)?
        }
    };
    assert_eq!(bytes.len(), uncompressed_length);
    Ok(bytes)
}

#[cfg(test)]
const GZIP_HEADER: [u8; 10] = [
    0x1f, 0x8b, // magic bytes
    8,    // using deflate
    0,    // no flags
    0, 0, 0, 0,   // timestamp = 0
    0,   // compression level (does not really matter)
    255, // OS set to 255 by default
];

/// writes zlib compressed bytes with a gzip header
#[cfg(test)]
pub(crate) fn write_gzip_compressed_bytes(
    output: &mut impl Write,
    bytes: &[u8],
    compression_level: u8,
) -> ReadResult<()> {
    output.write_all(GZIP_HEADER.as_slice())?;
    let compressed = miniz_oxide::deflate::compress_to_vec(bytes, compression_level);
    output.write_all(compressed.as_slice())?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_hierarchy_bytes(
    output: &mut (impl Write + Seek),
    compression: HierarchyCompression,
    bytes: &[u8],
) -> WriteResult<()> {
    // remember start to fix the section length afterwards
    let start = output.stream_position()?;
    write_u64(output, 0)?; // dummy section length
    let uncompressed_length = bytes.len() as u64;
    write_u64(output, uncompressed_length)?;

    match compression {
        HierarchyCompression::ZLib => {
            write_gzip_compressed_bytes(output, bytes, HIERARCHY_GZIP_COMPRESSION_LEVEL)?;
        }
        HierarchyCompression::Lz4 => {
            let compressed = lz4_flex::compress(bytes);
            output.write_all(&compressed)?;
        }
        HierarchyCompression::Lz4Duo => {
            let compressed_lvl1 = lz4_flex::compress(bytes);
            let lvl1_len = compressed_lvl1.len() as u64;
            write_variant_u64(output, lvl1_len)?;
            let compressed_lvl2 = lz4_flex::compress(&compressed_lvl1);
            output.write_all(&compressed_lvl2)?;
        }
    };

    // fix section length
    let end = output.stream_position()?;
    output.seek(SeekFrom::Start(start))?;
    write_u64(output, end - start)?;
    output.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn enum_table_from_string(value: String, handle: u64) -> ReadResult<FstHierarchyEntry> {
    let parts: Vec<&str> = value.split(' ').collect();
    if parts.len() < 2 {
        return Err(ReaderError::EnumTableString(
            "not enough spaces".to_string(),
            value,
        ));
    }
    let name = parts[0].to_string();
    let element_count = parts[1].parse::<usize>()?;
    let expected_part_len = element_count * 2;
    if parts.len() - 2 != expected_part_len {
        return Err(ReaderError::EnumTableString(
            format!(
                "expected {} parts got {}",
                expected_part_len,
                parts.len() - 2
            ),
            value,
        ));
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

#[cfg(test)]
fn enum_table_to_string(name: &str, mapping: &[(String, String)]) -> String {
    let mut out = String::with_capacity(name.len() + mapping.len() * 32 + 32);
    out.push_str(name);
    out.push(' ');
    out.push_str(&format!("{}", mapping.len()));
    for (_value, name) in mapping {
        out.push(' ');
        out.push_str(name);
    }
    for (value, _name) in mapping {
        out.push(' ');
        out.push_str(value);
    }
    out
}

const FST_SUP_VAR_DATA_TYPE_BITS: u32 = 10;
const FST_SUP_VAR_DATA_TYPE_MASK: u64 = (1 << FST_SUP_VAR_DATA_TYPE_BITS) - 1;

fn parse_misc_attribute(
    name: String,
    tpe: MiscType,
    arg: u64,
    arg2: Option<u64>,
) -> ReadResult<FstHierarchyEntry> {
    let res = match tpe {
        MiscType::Comment => FstHierarchyEntry::Comment { string: name },
        MiscType::EnvVar => todo!("EnvVar Attribute"), // fstWriterSetEnvVar()
        MiscType::SupVar => {
            // This attribute supplies VHDL specific information and is used by GHDL
            let var_type = (arg >> FST_SUP_VAR_DATA_TYPE_BITS) as u8;
            let data_type = (arg & FST_SUP_VAR_DATA_TYPE_MASK) as u8;
            FstHierarchyEntry::VhdlVarInfo {
                type_name: name,
                var_type: FstVhdlVarType::try_from_primitive(var_type)?,
                data_type: FstVhdlDataType::try_from_primitive(data_type)?,
            }
        }
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
        MiscType::ValueList => todo!("ValueList Attribute"), // fstWriterSetValueList()
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

fn read_hierarchy_attribute_arg2_encoded_as_name(input: &mut impl Read) -> ReadResult<u64> {
    let (value, _) = read_variant_u64(input)?;
    let end_byte = read_u8(input)?;
    assert_eq!(end_byte, 0, "expected to be zero terminated!");
    Ok(value)
}

const HIERARCHY_TPE_VCD_SCOPE: u8 = 254;
const HIERARCHY_TPE_VCD_UP_SCOPE: u8 = 255;
const HIERARCHY_TPE_VCD_ATTRIBUTE_BEGIN: u8 = 252;
const HIERARCHY_TPE_VCD_ATTRIBUTE_END: u8 = 253;

pub(crate) fn read_hierarchy_entry(
    input: &mut impl Read,
    handle_count: &mut u32,
) -> ReadResult<Option<FstHierarchyEntry>> {
    let entry_tpe = match read_u8(input) {
        Ok(tpe) => tpe,
        Err(_) => return Ok(None),
    };
    let entry = match entry_tpe {
        HIERARCHY_TPE_VCD_SCOPE => {
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
        HIERARCHY_TPE_VCD_UP_SCOPE => {
            // VcdUpScope (ScopeType)
            FstHierarchyEntry::UpScope
        }
        HIERARCHY_TPE_VCD_ATTRIBUTE_BEGIN => {
            let tpe = AttributeType::try_from_primitive(read_u8(input)?)?;
            let subtype = MiscType::try_from_primitive(read_u8(input)?)?;
            match tpe {
                AttributeType::Misc => {
                    let (name, arg2) = match subtype {
                        MiscType::SourceStem | MiscType::SourceInstantiationStem => {
                            let arg2 = read_hierarchy_attribute_arg2_encoded_as_name(input)?;
                            ("".to_string(), Some(arg2))
                        }
                        _ => {
                            let name = read_c_str(input, HIERARCHY_ATTRIBUTE_MAX_SIZE)?;
                            (name, None)
                        }
                    };
                    let (arg, _) = read_variant_u64(input)?;
                    parse_misc_attribute(name, subtype, arg, arg2)?
                }
                AttributeType::Array => todo!("ARRAY attributes"),
                AttributeType::Enum => todo!("ENUM attributes"),
                AttributeType::Pack => todo!("PACK attributes"),
            }
        }
        HIERARCHY_TPE_VCD_ATTRIBUTE_END => {
            // GenAttributeEnd (ScopeType)
            FstHierarchyEntry::AttributeEnd
        }

        other => todo!("Deal with hierarchy entry of type: {other}"),
    };

    Ok(Some(entry))
}

#[cfg(test)]
fn write_hierarchy_attribute(
    output: &mut impl Write,
    tpe: AttributeType,
    subtype: MiscType,
    name: &str,
    arg: u64,
    arg2: Option<u64>,
) -> WriteResult<()> {
    write_u8(output, HIERARCHY_TPE_VCD_ATTRIBUTE_BEGIN)?;
    write_u8(output, tpe as u8)?;
    write_u8(output, subtype as u8)?;
    let raw_name_bytes = match arg2 {
        None => {
            assert!(name.len() <= HIERARCHY_ATTRIBUTE_MAX_SIZE);
            name.to_string().into_bytes()
        }
        Some(value) => {
            assert!(name.is_empty(), "cannot have a name + an arg2!");
            let mut buf = vec![0u8; 10];
            let mut buf_writer: &mut [u8] = buf.as_mut();
            let len = write_variant_u64(&mut buf_writer, value)?;
            buf.truncate(len);
            buf
        }
    };
    output.write_all(&raw_name_bytes)?;
    write_u8(output, 0)?; // zero terminate string/variant
    write_variant_u64(output, arg)?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_hierarchy_entry(
    output: &mut impl Write,
    handle_count: &mut u32,
    entry: &FstHierarchyEntry,
) -> WriteResult<()> {
    match entry {
        FstHierarchyEntry::Scope {
            tpe,
            name,
            component,
        } => {
            write_u8(output, HIERARCHY_TPE_VCD_SCOPE)?;
            write_u8(output, *tpe as u8)?;
            assert!(name.len() <= HIERARCHY_NAME_MAX_SIZE);
            write_c_str(output, name)?;
            assert!(component.len() <= HIERARCHY_NAME_MAX_SIZE);
            write_c_str(output, component)?;
        }
        FstHierarchyEntry::UpScope => {
            write_u8(output, HIERARCHY_TPE_VCD_UP_SCOPE)?;
        }
        FstHierarchyEntry::Var {
            tpe,
            direction,
            name,
            length,
            handle,
            is_alias,
        } => {
            write_u8(output, *tpe as u8)?;
            write_u8(output, *direction as u8)?;
            assert!(name.len() <= HIERARCHY_NAME_MAX_SIZE);
            write_c_str(output, name)?;
            let raw_length = if *tpe == FstVarType::Port {
                3 * (*length) + 2
            } else {
                *length
            };
            write_variant_u32(output, raw_length)?;
            if *is_alias {
                write_variant_u32(output, handle.get_raw())?;
            } else {
                // sanity check handle
                assert_eq!(handle.get_index(), *handle_count as usize);
                *handle_count += 1;
                // write no-alias
                write_variant_u32(output, 0)?;
            }
        }
        FstHierarchyEntry::PathName { name, id } => write_hierarchy_attribute(
            output,
            AttributeType::Misc,
            MiscType::PathName,
            name,
            *id,
            None,
        )?,
        FstHierarchyEntry::SourceStem {
            is_instantiation,
            path_id,
            line,
        } => {
            let subtpe = if *is_instantiation {
                MiscType::SourceInstantiationStem
            } else {
                MiscType::SourceStem
            };
            write_hierarchy_attribute(
                output,
                AttributeType::Misc,
                subtpe,
                "",
                *line,
                Some(*path_id),
            )?
        }
        FstHierarchyEntry::Comment { string } => write_hierarchy_attribute(
            output,
            AttributeType::Misc,
            MiscType::Comment,
            string,
            0,
            None,
        )?,
        FstHierarchyEntry::EnumTable {
            name,
            handle,
            mapping,
        } => {
            let table_str = enum_table_to_string(name, mapping);
            write_hierarchy_attribute(
                output,
                AttributeType::Misc,
                MiscType::EnumTable,
                &table_str,
                *handle,
                None,
            )?
        }
        FstHierarchyEntry::EnumTableRef { handle } => write_hierarchy_attribute(
            output,
            AttributeType::Misc,
            MiscType::EnumTable,
            "",
            *handle,
            None,
        )?,
        FstHierarchyEntry::VhdlVarInfo {
            type_name,
            var_type,
            data_type,
        } => {
            let arg = ((*var_type as u64) << FST_SUP_VAR_DATA_TYPE_BITS) | (*data_type as u64);
            write_hierarchy_attribute(
                output,
                AttributeType::Misc,
                MiscType::SupVar,
                type_name,
                arg,
                None,
            )?;
        }
        FstHierarchyEntry::AttributeEnd => {
            write_u8(output, HIERARCHY_TPE_VCD_ATTRIBUTE_END)?;
        }
    }

    Ok(())
}

//////////////// Vale Change Data

pub(crate) fn read_packed_signal_value_bytes(
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
            ValueChangePackType::FastLz => {
                let compressed_length = (len - skiplen) as u64;
                crate::fastlz::decompress(
                    input,
                    compressed_length as usize,
                    uncompressed_length as usize,
                )?
            }
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

pub(crate) fn read_time_table(
    input: &mut (impl Read + Seek),
    section_start: u64,
    section_length: u64,
) -> ReadResult<(u64, Vec<u64>)> {
    // the time block meta data is in the last 24 bytes at the end of the section
    input.seek(SeekFrom::Start(section_start + section_length - 3 * 8))?;
    let uncompressed_length = read_u64(input)?;
    let compressed_length = read_u64(input)?;
    let number_of_items = read_u64(input)?;
    assert!(compressed_length <= section_length);

    // now that we know how long the block actually is, we can go back to it
    input.seek(SeekFrom::Current(-(3 * 8) - (compressed_length as i64)))?;
    let bytes = read_zlib_compressed_bytes(input, uncompressed_length, compressed_length, true)?;
    let mut byte_reader: &[u8] = &bytes;
    let mut time_table: Vec<u64> = Vec::with_capacity(number_of_items as usize);
    let mut time_val: u64 = 0; // running time counter

    for _ in 0..number_of_items {
        let (value, _) = read_variant_u64(&mut byte_reader)?;
        time_val += value;
        time_table.push(time_val);
    }

    let time_section_length = compressed_length + 3 * 8;
    Ok((time_section_length, time_table))
}

#[cfg(test)]
pub(crate) fn write_time_table(
    output: &mut (impl Write + Seek),
    compression: Option<u8>,
    table: &[u64],
) -> WriteResult<()> {
    // delta compress
    let num_entries = table.len();
    let table = delta_compress_time_table(table)?;
    // write data
    let (uncompressed_len, compressed_len) = match compression {
        Some(comp) => {
            let compressed = miniz_oxide::deflate::compress_to_vec_zlib(table.as_slice(), comp);
            // is compression worth it?
            if compressed.len() < table.len() {
                output.write_all(compressed.as_slice())?;
                (table.len(), compressed.len())
            } else {
                // it is more space efficient to stick with the uncompressed version
                output.write_all(table.as_slice())?;
                (table.len(), table.len())
            }
        }
        None => {
            output.write_all(table.as_slice())?;
            (table.len(), table.len())
        }
    };
    write_u64(output, uncompressed_len as u64)?;
    write_u64(output, compressed_len as u64)?;
    write_u64(output, num_entries as u64)?;

    Ok(())
}

#[cfg(test)]
#[inline]
fn delta_compress_time_table(table: &[u64]) -> WriteResult<Vec<u8>> {
    let mut output = vec![];
    let mut prev_time = 0u64;
    for time in table {
        let delta = *time - prev_time;
        prev_time = *time;
        write_variant_u64(&mut output, delta)?;
    }
    Ok(output)
}
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn read_frame(
    input: &mut (impl Read + Seek),
    section_start: u64,
    section_length: u64,
    signals: &[SignalInfo],
    signal_filter: &BitMask,
    float_endian: FloatingPointEndian,
    start_time: u64,
    callback: &mut impl FnMut(u64, FstSignalHandle, FstSignalValue),
) -> ReadResult<()> {
    // we skip the section header (section_length, start_time, end_time, ???)
    input.seek(SeekFrom::Start(section_start + 4 * 8))?;
    let (uncompressed_length, _) = read_variant_u64(input)?;
    let (compressed_length, _) = read_variant_u64(input)?;
    let (max_handle, _) = read_variant_u64(input)?;
    assert!(compressed_length <= section_length);
    let bytes_vec =
        read_zlib_compressed_bytes(input, uncompressed_length, compressed_length, true)?;
    let mut bytes = std::io::Cursor::new(bytes_vec);

    assert_eq!(signals.len(), max_handle as usize);
    for (idx, signal) in signals.iter().enumerate() {
        let signal_length = signal.len();
        if signal_filter.is_set(idx) {
            let handle = FstSignalHandle::from_index(idx);
            match signal_length {
                0 => {} // ignore since variable-length records have no initial value
                len => {
                    if !signal.is_real() {
                        let value = read_bytes(&mut bytes, len as usize)?;
                        callback(start_time, handle, FstSignalValue::String(&value));
                    } else {
                        let value = read_f64(&mut bytes, float_endian)?;
                        callback(start_time, handle, FstSignalValue::Real(value));
                    }
                }
            }
        } else {
            // skip
            bytes.seek(SeekFrom::Current(signal_length as i64))?;
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn skip_frame(input: &mut (impl Read + Seek), section_start: u64) -> ReadResult<()> {
    // we skip the section header (section_length, start_time, end_time, ???)
    input.seek(SeekFrom::Start(section_start + 4 * 8))?;
    let (_uncompressed_length, _) = read_variant_u64(input)?;
    let (compressed_length, _) = read_variant_u64(input)?;
    let (_max_handle, _) = read_variant_u64(input)?;
    input.seek(SeekFrom::Current(compressed_length as i64))?;
    Ok(())
}

/// Table of signal offsets inside a data block.
#[derive(Debug)]
pub(crate) struct OffsetTable(Vec<SignalDataLoc>);

impl From<Vec<SignalDataLoc>> for OffsetTable {
    fn from(value: Vec<SignalDataLoc>) -> Self {
        Self(value)
    }
}

impl OffsetTable {
    pub(crate) fn iter(&self) -> OffsetTableIter {
        OffsetTableIter {
            table: self,
            signal_idx: 0,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    fn get_entry(&self, signal_idx: usize) -> Option<OffsetEntry> {
        match &self.0[signal_idx] {
            SignalDataLoc::None => None,
            // aliases should always directly point to an offset,
            // so we should not have to recurse!
            SignalDataLoc::Alias(alias_idx) => match &self.0[*alias_idx as usize] {
                SignalDataLoc::Offset(offset, len) => Some(OffsetEntry {
                    signal_idx,
                    offset: offset.get() as u64,
                    len: len.get(),
                }),
                _ => unreachable!("aliases should always directly point to an offset"),
            },
            SignalDataLoc::Offset(offset, len) => Some(OffsetEntry {
                signal_idx,
                offset: offset.get() as u64,
                len: len.get(),
            }),
        }
    }
}

pub(crate) struct OffsetTableIter<'a> {
    table: &'a OffsetTable,
    signal_idx: usize,
}

#[derive(Debug)]
pub(crate) struct OffsetEntry {
    pub(crate) signal_idx: usize,
    pub(crate) offset: u64,
    pub(crate) len: u32,
}
impl Iterator for OffsetTableIter<'_> {
    type Item = OffsetEntry;

    fn next(&mut self) -> Option<Self::Item> {
        // get the first entry which is not None
        while self.signal_idx < self.table.0.len()
            && matches!(self.table.0[self.signal_idx], SignalDataLoc::None)
        {
            self.signal_idx += 1
        }

        // did we reach the end?
        if self.signal_idx >= self.table.0.len() {
            return None;
        }

        // increment id for next call
        self.signal_idx += 1;

        // return result
        let res = self.table.get_entry(self.signal_idx - 1);
        debug_assert!(res.is_some());
        res
    }
}

fn read_value_change_alias2(
    mut chain_bytes: &[u8],
    max_handle: u64,
    last_table_entry: u32,
) -> ReadResult<OffsetTable> {
    let mut table = Vec::with_capacity(max_handle as usize);
    let mut offset: Option<NonZeroU32> = None;
    let mut prev_alias = 0u32;
    let mut prev_offset_idx = 0usize;
    while !chain_bytes.is_empty() {
        let idx = table.len();
        let kind = chain_bytes[0];
        if (kind & 1) == 1 {
            let shval = read_variant_i64(&mut chain_bytes)? >> 1;
            match shval.cmp(&0) {
                Ordering::Greater => {
                    // a new incremental offset
                    let new_offset = NonZeroU32::new(
                        (offset.map(|o| o.get()).unwrap_or_default() as i64 + shval) as u32,
                    )
                    .unwrap();
                    // if there was a previous entry, we need to update the length
                    if let Some(prev_offset) = offset {
                        let len = NonZeroU32::new(new_offset.get() - prev_offset.get()).unwrap();
                        table[prev_offset_idx] = SignalDataLoc::Offset(prev_offset, len);
                    }
                    offset = Some(new_offset);
                    prev_offset_idx = idx;
                    // push a placeholder which will be replaced as soon as we know the length
                    table.push(SignalDataLoc::None);
                }
                Ordering::Less => {
                    // new signal alias
                    prev_alias = (-shval - 1) as u32;
                    table.push(SignalDataLoc::Alias(prev_alias));
                }
                Ordering::Equal => {
                    // same signal alias as previous signal
                    table.push(SignalDataLoc::Alias(prev_alias));
                }
            }
        } else {
            // a block of signals that do not have any data
            let (value, _) = read_variant_u32(&mut chain_bytes)?;
            let zeros = value >> 1;
            for _ in 0..zeros {
                table.push(SignalDataLoc::None);
            }
        }
    }

    // if there was a previous entry, we need to update the length
    if let Some(prev_offset) = offset {
        let len = NonZeroU32::new(last_table_entry - prev_offset.get()).unwrap();
        table[prev_offset_idx] = SignalDataLoc::Offset(prev_offset, len);
    }

    Ok(table.into())
}

fn read_value_change_alias(
    mut chain_bytes: &[u8],
    max_handle: u64,
    last_table_entry: u32,
) -> ReadResult<OffsetTable> {
    let mut table = Vec::with_capacity(max_handle as usize);
    let mut prev_offset_idx = 0usize;
    let mut offset: Option<NonZeroU32> = None;
    while !chain_bytes.is_empty() {
        let (raw_val, _) = read_variant_u32(&mut chain_bytes)?;
        let idx = table.len();
        if raw_val == 0 {
            let (raw_alias, _) = read_variant_u32(&mut chain_bytes)?;
            let alias = ((raw_alias as i64) - 1) as u32;
            table.push(SignalDataLoc::Alias(alias));
        } else if (raw_val & 1) == 1 {
            // a new incremental offset
            let new_offset =
                NonZeroU32::new(offset.map(|o| o.get()).unwrap_or_default() + (raw_val >> 1))
                    .unwrap();
            // if there was a previous entry, we need to update the length
            if let Some(prev_offset) = offset {
                let len = NonZeroU32::new(new_offset.get() - prev_offset.get()).unwrap();
                table[prev_offset_idx] = SignalDataLoc::Offset(prev_offset, len);
            }
            offset = Some(new_offset);
            prev_offset_idx = idx;
            // push a placeholder which will be replaced as soon as we know the length
            table.push(SignalDataLoc::None);
        } else {
            // a block of signals that do not have any data
            let zeros = raw_val >> 1;
            for _ in 0..zeros {
                table.push(SignalDataLoc::None);
            }
        }
    }

    // if there was a previous entry, we need to update the length
    if let Some(prev_offset) = offset {
        let len = NonZeroU32::new(last_table_entry - prev_offset.get()).unwrap();
        table[prev_offset_idx] = SignalDataLoc::Offset(prev_offset, len);
    }

    Ok(table.into())
}

/// Indicates the location of the signal data for the current block.
#[derive(Debug, Copy, Clone)]
enum SignalDataLoc {
    /// The signal has no value changes in the current block.
    None,
    /// The signal has the same offset as another signal.
    Alias(u32),
    /// The signal has a new offset.
    Offset(NonZeroU32, NonZeroU32),
}

pub(crate) fn read_signal_locs(
    input: &mut (impl Read + Seek),
    chain_len_offset: u64,
    section_kind: DataSectionKind,
    max_handle: u64,
    start: u64,
) -> ReadResult<OffsetTable> {
    input.seek(SeekFrom::Start(chain_len_offset))?;
    let chain_compressed_length = read_u64(input)?;

    // the chain starts _chain_length_ bytes before the chain length
    let chain_start = chain_len_offset - chain_compressed_length;
    input.seek(SeekFrom::Start(chain_start))?;
    let chain_bytes = read_bytes(input, chain_compressed_length as usize)?;

    let last_table_entry = (chain_start - start) as u32; // indx_pos - vc_start
    if section_kind == DataSectionKind::DynamicAlias2 {
        read_value_change_alias2(&chain_bytes, max_handle, last_table_entry)
    } else {
        read_value_change_alias(&chain_bytes, max_handle, last_table_entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn data_struct_sizes() {
        assert_eq!(
            std::mem::size_of::<SignalDataLoc>(),
            std::mem::size_of::<u64>() + std::mem::size_of::<u32>()
        );
    }

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
    fn regression_test_read_write_variant_i64() {
        do_test_read_write_variant_i64(-36028797018963969);
        do_test_read_write_variant_i64(-4611686018427387905);
    }

    fn do_test_read_write_variant_i64(value: i64) {
        let mut buf = std::io::Cursor::new(vec![0u8; 24]);
        write_variant_i64(&mut buf, value).unwrap();
        buf.seek(SeekFrom::Start(0)).unwrap();
        let read_value = read_variant_i64(&mut buf).unwrap();
        assert_eq!(read_value, value);
    }

    proptest! {
         #[test]
        fn test_read_write_variant_u64(value: u64) {
            let mut buf = std::io::Cursor::new(vec![0u8; 24]);
            write_variant_u64(&mut buf, value).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            let (read_value, _) = read_variant_u64(&mut buf).unwrap();
            assert_eq!(read_value, value);
        }

         #[test]
        fn test_read_write_variant_i64(value: i64) {
            do_test_read_write_variant_i64(value);
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

    /// makes sure that there are no zero bytes inside the string and that the max length is obeyed
    fn is_valid_c_str(value: &str, max_len: usize) -> bool {
        let string_bytes: &[u8] = value.as_bytes();
        let len_constraint = string_bytes.len() < max_len;
        let non_zero_constraint = !string_bytes.contains(&0u8);
        len_constraint && non_zero_constraint
    }

    fn is_valid_alphanumeric_c_str(value: &str, max_len: usize) -> bool {
        let alphanumeric_constraint = value.chars().all(|c| c.is_alphanumeric());
        is_valid_c_str(value, max_len) && alphanumeric_constraint
    }

    proptest! {
        #[test]
        fn test_write_c_str_fixed_length(string: String, max_len in 1 .. 400usize) {
            prop_assume!(is_valid_c_str(&string, max_len));
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
        fn test_write_c_str(string: String, max_len in 1 .. 400usize) {
            prop_assume!(is_valid_c_str(&string, max_len));
            let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
            write_c_str(&mut buf, &string).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            assert_eq!(
                read_c_str(&mut buf, max_len).unwrap(),
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
            let compressed_len = write_compressed_bytes(&mut buf, &bytes, 3, allow_uncompressed).unwrap();
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
        fn test_read_write_blackout(mut blackouts: Vec<BlackoutData>) {
            // blackout times must be in increasing order => sort
            blackouts.sort_by(|a, b| a.time.cmp(&b.time));

            // actual test
            let max_len = blackouts.len() * 5 + 3 * 8;
            let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
            write_blackout(&mut buf, &blackouts).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            let actual = read_blackout(&mut buf).unwrap();
            assert_eq!(actual.len(), blackouts.len());
            assert_eq!(actual, blackouts);
        }
    }

    proptest! {
        #[test]
        fn test_read_write_geometry(signals: Vec<SignalInfo>) {
            let max_len = signals.len() * 4 + 3 * 8;
            let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
            write_geometry(&mut buf, &signals, 3).unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            let actual = read_geometry(&mut buf).unwrap();
            assert_eq!(actual.len(), signals.len());
            assert_eq!(actual, signals);
        }
    }

    /// ensures that no string contains zero bytes or is longer than max_len
    fn hierarchy_entry_with_valid_c_strings(entry: &FstHierarchyEntry) -> bool {
        match entry {
            FstHierarchyEntry::Scope {
                name, component, ..
            } => {
                is_valid_c_str(name, HIERARCHY_NAME_MAX_SIZE)
                    && is_valid_c_str(component, HIERARCHY_NAME_MAX_SIZE)
            }
            FstHierarchyEntry::UpScope => true,
            FstHierarchyEntry::Var { name, .. } => is_valid_c_str(name, HIERARCHY_NAME_MAX_SIZE),
            FstHierarchyEntry::PathName { name, .. } => {
                is_valid_c_str(name, HIERARCHY_ATTRIBUTE_MAX_SIZE)
            }
            FstHierarchyEntry::SourceStem { .. } => true,
            FstHierarchyEntry::Comment { string } => {
                is_valid_c_str(string, HIERARCHY_ATTRIBUTE_MAX_SIZE)
            }
            FstHierarchyEntry::EnumTable { name, mapping, .. } => {
                is_valid_alphanumeric_c_str(name, HIERARCHY_ATTRIBUTE_MAX_SIZE)
                    && mapping.iter().all(|(k, v)| {
                        is_valid_alphanumeric_c_str(k, HIERARCHY_ATTRIBUTE_MAX_SIZE)
                            && is_valid_alphanumeric_c_str(v, HIERARCHY_ATTRIBUTE_MAX_SIZE)
                    })
            }
            FstHierarchyEntry::EnumTableRef { .. } => true,
            FstHierarchyEntry::VhdlVarInfo { type_name, .. } => {
                is_valid_c_str(type_name, HIERARCHY_NAME_MAX_SIZE)
            }
            FstHierarchyEntry::AttributeEnd => true,
        }
    }

    /// ensures that the mapping strings are non-empty and do not contain spaces
    fn hierarchy_entry_with_valid_mapping(entry: &FstHierarchyEntry) -> bool {
        match entry {
            FstHierarchyEntry::EnumTable { mapping, .. } => mapping
                .iter()
                .all(|(k, v)| is_valid_mapping_str(k) && is_valid_mapping_str(v)),
            _ => true,
        }
    }
    fn is_valid_mapping_str(value: &str) -> bool {
        !value.is_empty() && !value.contains(' ')
    }

    /// ensures that ports are not too wide
    fn hierarchy_entry_with_valid_port_width(entry: &FstHierarchyEntry) -> bool {
        if let FstHierarchyEntry::Var {
            tpe: FstVarType::Port,
            length,
            ..
        } = entry
        {
            *length < (u32::MAX / 3) - 2
        } else {
            true
        }
    }

    fn read_write_hierarchy_entry(entry: FstHierarchyEntry) {
        // the handle count is only important if we are writing a non-aliased variable
        let base_handle_count: u32 = match &entry {
            FstHierarchyEntry::Var {
                handle, is_alias, ..
            } => {
                if *is_alias {
                    0
                } else {
                    handle.get_index() as u32
                }
            }
            _ => 0,
        };

        let max_len = 1024 * 64;
        let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
        let mut handle_count = base_handle_count;
        write_hierarchy_entry(&mut buf, &mut handle_count, &entry).unwrap();
        if base_handle_count > 0 {
            assert_eq!(handle_count, base_handle_count + 1);
        }
        buf.seek(SeekFrom::Start(0)).unwrap();
        handle_count = base_handle_count;
        let actual = read_hierarchy_entry(&mut buf, &mut handle_count)
            .unwrap()
            .unwrap();
        assert_eq!(actual, entry);
    }

    #[test]
    fn test_read_write_hierarchy_path_name_entry() {
        let entry = FstHierarchyEntry::PathName {
            id: 1,
            name: "".to_string(),
        };
        read_write_hierarchy_entry(entry);
    }

    proptest! {
        #[test]
        fn test_prop_read_write_hierarchy_entry(entry: FstHierarchyEntry) {
            prop_assume!(hierarchy_entry_with_valid_c_strings(&entry));
            prop_assume!(hierarchy_entry_with_valid_mapping(&entry));
            prop_assume!(hierarchy_entry_with_valid_port_width(&entry));
            read_write_hierarchy_entry(entry);
        }
    }

    // test with some manually chosen entries
    #[test]
    fn test_read_write_hierarchy_entry() {
        // make sure that we can write and read long attributes
        let entry = FstHierarchyEntry::Comment {
            string: "TEST ".repeat((8000 + 4) / 5),
        };
        read_write_hierarchy_entry(entry);
    }

    fn do_test_read_write_hierarchy_bytes(tpe: HierarchyCompression, bytes: Vec<u8>) {
        let max_len = std::cmp::max(64, bytes.len() + 3 * 8);
        let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
        write_hierarchy_bytes(&mut buf, tpe, &bytes).unwrap();
        buf.seek(SeekFrom::Start(0)).unwrap();
        let actual = read_hierarchy_bytes(&mut buf, tpe).unwrap();
        assert_eq!(actual, bytes);
    }

    #[test]
    fn test_read_write_hierarchy_bytes_regression() {
        do_test_read_write_hierarchy_bytes(HierarchyCompression::Lz4, vec![]);
        do_test_read_write_hierarchy_bytes(HierarchyCompression::ZLib, vec![]);
    }

    proptest! {
        #[test]
        fn test_prop_read_write_hierarchy_bytes(tpe: HierarchyCompression, bytes: Vec<u8>) {
            do_test_read_write_hierarchy_bytes(tpe, bytes);
        }
    }

    fn read_write_time_table(mut table: Vec<u64>, compressed: bool) {
        // the table has to be sorted since we are computing and saving time deltas
        table.sort();
        let max_len = std::cmp::max(64, table.len() * 8 + 3 * 8);
        let mut buf = std::io::Cursor::new(vec![0u8; max_len]);
        let comp = if compressed { Some(3) } else { None };
        write_time_table(&mut buf, comp, &table).unwrap();
        let section_start = 0u64;
        let section_length = buf.stream_position().unwrap();
        buf.seek(SeekFrom::Start(0)).unwrap();
        let (actual_len, actual_table) =
            read_time_table(&mut buf, section_start, section_length).unwrap();
        assert_eq!(actual_len, section_length);
        assert_eq!(actual_table, table);
    }

    #[test]
    fn test_read_write_time_table_uncompressed() {
        let table = vec![1, 0];
        read_write_time_table(table, false);
    }

    #[test]
    fn test_read_write_time_table_compressed() {
        let table = (0..10000).collect();
        read_write_time_table(table, true);
    }

    proptest! {
        #[test]
        fn test_prop_read_write_time_table(table: Vec<u64>, compressed: bool) {
            read_write_time_table(table, compressed);
        }
    }
}
