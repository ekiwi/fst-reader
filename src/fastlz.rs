// Copyright 2023 The Regents of the University of California
// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// Simple Rust implementation of FastLZ: https://github.com/ariya/FastLZ
// Currently only reading is supported!

use crate::io::{read_bytes, read_u8, ReadResult};
use std::io::{Read, Seek, SeekFrom};

pub(crate) fn decompress(
    input: &mut (impl Read + Seek),
    input_len: usize,
    output_size_hint: usize,
) -> ReadResult<Vec<u8>> {
    let mut out = Vec::with_capacity(output_size_hint);

    let header = read_u8(input)?;
    let level = (header >> 5) + 1;
    // go back to header which is actually the first op code!
    input.seek(SeekFrom::Current(-1))?;

    match level {
        1 => decompress_level1(input, input_len, &mut out)?,
        2 => decompress_level2(input, input_len, &mut out)?,
        other => todo!("Better error handling for invalid fastlz level {other}!"),
    };
    Ok(out)
}

fn decompress_level1(input: &mut impl Read, input_len: usize, out: &mut Vec<u8>) -> ReadResult<()> {
    let mut read_count: usize = 0;

    while read_count < input_len {
        let byte0 = read_u8(input)?;
        read_count += 1;
        // long or short match
        if byte0 >= 32 {
            let mut length = (byte0 >> 5) as usize + 2;
            let offset = 256 * ((byte0 & 0x1f) as usize);
            let mut start = out.len() - offset - 1;
            // long run (i.e. type == 7)
            if length == 7 + 2 {
                length += read_u8(input)? as usize;
                read_count += 1;
            }
            start -= read_u8(input)? as usize; // offset adjustment
            read_count += 1;
            copy_match(out, start, length);
        } else {
            literal_run(input, byte0, &mut read_count, out)?;
        }
    }
    Ok(())
}

const MAX_L2_DISTANCE: usize = 8191;

fn decompress_level2(input: &mut impl Read, input_len: usize, out: &mut Vec<u8>) -> ReadResult<()> {
    let mut read_count: usize = 0;
    let mut byte0 = read_u8(input)? & 0x1f; // remove header for first read
    read_count += 1;

    loop {
        // long or short match
        if byte0 >= 32 {
            let mut length = (byte0 >> 5) as usize + 2;
            let offset = 256 * ((byte0 & 0x1f) as usize);
            let mut start = out.len() - offset - 1;
            // long run (i.e. type == 7)
            if length == 7 + 2 {
                // lvl 2: read length until we get to a non 0xff byte
                loop {
                    let code = read_u8(input)?;
                    read_count += 1;
                    length += code as usize;
                    if code != 255 {
                        break;
                    }
                }
            }
            let offset_code = read_u8(input)?;
            read_count += 1;
            start -= offset_code as usize; // offset adjustment
                                           // lvl 2: match from 16-bit distance
            if offset_code == 255 && byte0 & 0x1f == 31 {
                let lvl2_offset_high = (read_u8(input)? as usize) << 8;
                let lvl2_offset = lvl2_offset_high + read_u8(input)? as usize;
                read_count += 2;
                // overwrite start
                start = out.len() - lvl2_offset - MAX_L2_DISTANCE - 1;
            }
            copy_match(out, start, length);
        } else {
            literal_run(input, byte0, &mut read_count, out)?;
        }

        // exit the loop
        if read_count >= input_len {
            break;
        }

        // load next instruction
        byte0 = read_u8(input)?;
        read_count += 1;
    }
    Ok(())
}

#[inline]
fn literal_run(
    input: &mut impl Read,
    byte0: u8,
    read_count: &mut usize,
    out: &mut Vec<u8>,
) -> ReadResult<()> {
    let run_length = (1 + byte0) as usize;
    let mut bytes = read_bytes(input, run_length)?;
    *read_count += run_length;
    out.append(&mut bytes);
    Ok(())
}

#[inline]
fn copy_match(out: &mut Vec<u8>, start: usize, length: usize) {
    for ii in start..start + length {
        out.push(out[ii]);
    }
}
