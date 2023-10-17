// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
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
    match level {
        1 => {
            // go back to header which is actually the first op code!
            input.seek(SeekFrom::Current(-1))?;
            decompress_level1(input, input_len - 1, &mut out)?
        }
        2 => todo!("implement level2 support"),
        other => todo!("Better error handling for invalid fastlz level {other}!"),
    };
    Ok(out)
}

fn decompress_level1(input: &mut impl Read, input_len: usize, out: &mut Vec<u8>) -> ReadResult<()> {
    let mut read_count: usize = 0;

    while read_count < input_len {
        let byte0 = read_u8(input)?;
        read_count += 1;
        let tpe = byte0 >> 5;
        if tpe == 0 {
            // literal run
            let run_length = (1 + byte0) as usize;
            let mut bytes = read_bytes(input, run_length)?;
            read_count += run_length;
            out.append(&mut bytes);
        } else if tpe < 7 {
            // short match
            let byte1 = read_u8(input)?;
            read_count += 1;
            let offset = 256 * ((byte0 & 0x1f) as usize) + byte1 as usize;
            let length = 2 + tpe as usize;
            copy_match(out, offset, length);
        } else {
            // long match
            let byte1 = read_u8(input)?;
            let byte2 = read_u8(input)?;
            read_count += 2;
            let offset = 256 * ((byte0 & 0x1f) as usize) + byte2 as usize;
            let length = 9 + byte1 as usize;
            copy_match(out, offset, length);
        }
    }
    Ok(())
}

#[inline]
fn copy_match(out: &mut Vec<u8>, offset: usize, length: usize) {
    let start = out.len() - offset - 1;
    for ii in start..start + length {
        out.push(out[ii]);
    }
}
