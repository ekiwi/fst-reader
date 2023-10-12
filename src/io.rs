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

pub type ReadResult<T> = std::result::Result<T, ReaderError>;

pub type WriteResult<T> = std::result::Result<T, ReaderError>;

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

pub(crate) fn read_compressed_bytes(
    input: &mut (impl Read + Seek),
    uncompressed_length: u64,
    compressed_length: u64,
    skip_equal_len: bool,
) -> ReadResult<Vec<u8>> {
    let bytes = if uncompressed_length == compressed_length && skip_equal_len {
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

#[inline]
pub(crate) fn read_bytes(input: &mut impl Read, len: usize) -> ReadResult<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    input.take(len as u64).read_to_end(&mut buf)?;
    Ok(buf)
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

    #[test]
    fn test_write_c_str_fixed_length() {
        let mut buf = [0u8; 128];
        write_c_str_fixed_length(&mut buf.as_mut(), "test", 100).unwrap();
        assert_eq!(
            read_c_str_fixed_length(&mut buf.as_slice(), 100).unwrap(),
            "test"
        );

        let s2 = "hb42u9423yv324g2396v@#$----   23rf327845327";
        write_c_str_fixed_length(&mut buf.as_mut(), s2, 100).unwrap();
        assert_eq!(
            read_c_str_fixed_length(&mut buf.as_slice(), 100).unwrap(),
            s2
        );
    }

    #[test]
    fn test_read_write_header() {
        let header = Header {
            start_time: 13478,
            end_time: 12374738694,
            memory_used_by_writer: 59374829374,
            scope_count: 47321896453468,
            var_count: 4671823496,
            max_var_id_code: 4328947203984,
            vc_section_count: 324782364783264,
            timescale_exponent: -5,
            version: "brh2u39   - --  ÖÖÖÄÄr273g4923g4".to_string(),
            date: "123ß25434ß434324-----32421".to_string(),
            file_type: FileType::Verilog,
            time_zero: 123,
        };
        let mut buf = [0u8; 512];
        write_header(&mut buf.as_mut(), &header).unwrap();
        let (actual_header, endian) = read_header(&mut buf.as_slice()).unwrap();
        assert_eq!(endian, FloatingPointEndian::Little);
        assert_eq!(actual_header, header);
    }
}
