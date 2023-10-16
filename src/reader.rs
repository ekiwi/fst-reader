// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::io::*;
use crate::types::*;
use std::cmp::Ordering;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};

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
        callback: impl FnMut(u64, FstSignalHandle, FstSignalValue),
    ) -> Result<()> {
        // convert user filters
        let signal_count = self.meta.signals.len();
        let signal_mask = if let Some(signals) = &filter.include {
            let mut signal_mask = BitMask::repeat(false, signal_count);
            for sig in signals {
                let signal_idx = sig.get_index();
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

pub enum FstSignalValue<'a> {
    String(&'a str),
    Real(f64),
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
    mut callback: impl FnMut(u64, FstSignalHandle, FstSignalValue),
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
        Ok(None)
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
            target.write_all(&buf[..read_len])?;
        }
        // go to start of new file and return
        target.seek(SeekFrom::Start(0))?;
        let new_input = std::io::BufReader::new(target);
        Ok(Some(new_input))
    }
}

#[derive(Debug)]
struct MetaData {
    header: Header,
    signals: Vec<SignalInfo>,
    data_sections: Vec<DataSectionInfo>,
    float_endian: FloatingPointEndian,
    hierarchy_compression: HierarchyCompression,
    hierarchy_offset: u64,
}

pub type Result<T> = std::result::Result<T, ReaderError>;

struct HeaderReader<R: Read + Seek> {
    input: R,
    header: Option<Header>,
    signals: Option<Vec<SignalInfo>>,
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
        // skip this section if the header is not complete
        if self.header_incomplete() {
            let section_length = read_u64(&mut self.input)?;
            self.skip(section_length, 8)?;
        } else {
            self.signals = Some(read_geometry(&mut self.input)?);
        }
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
                    kind: ReaderErrorKind::IO(_),
                }) => {
                    break;
                }
                Err(other) => return Err(other),
                Ok(tpe) => tpe,
            };
            match block_tpe {
                BlockType::Header => {
                    let (header, endian) = read_header(&mut self.input)?;
                    self.header = Some(header);
                    self.float_endian = endian;
                }
                BlockType::VcData => self.read_data(&block_tpe)?,
                BlockType::VcDataDynamicAlias => self.read_data(&block_tpe)?,
                BlockType::VcDataDynamicAlias2 => self.read_data(&block_tpe)?,
                BlockType::Blackout => todo!("blackout"),
                BlockType::Geometry => self.read_geometry()?,
                BlockType::Hierarchy => self.read_hierarchy(HierarchyCompression::ZLib)?,
                BlockType::HierarchyLZ4 => self.read_hierarchy(HierarchyCompression::Lz4)?,
                BlockType::HierarchyLZ4Duo => self.read_hierarchy(HierarchyCompression::Lz4Duo)?,
                BlockType::GZipWrapper => panic!("GZip Wrapper should have been handled earlier!"),
                BlockType::Skip => todo!("skip block"),
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

struct DataReader<'a, R: Read + Seek, F: FnMut(u64, FstSignalHandle, FstSignalValue)> {
    input: &'a mut R,
    meta: &'a MetaData,
    filter: &'a DataFilter,
    callback: &'a mut F,
}

#[inline]
fn push_zeros(chain_table: &mut Vec<i64>, zeros: u32) {
    for _ in 0..zeros {
        chain_table.push(0);
    }
}

impl<'a, R: Read + Seek, F: FnMut(u64, FstSignalHandle, FstSignalValue)> DataReader<'a, R, F> {
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
        let bytes = read_zlib_compressed_bytes(
            &mut self.input,
            uncompressed_length,
            compressed_length,
            true,
        )?;
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
        let bytes = read_zlib_compressed_bytes(
            &mut self.input,
            uncompressed_length,
            compressed_length,
            true,
        )?;

        let mut byte_reader: &[u8] = &bytes;
        for idx in 0..(max_handle as usize) {
            let signal_length = self.meta.signals[idx].len();
            if self.filter.signals[idx] {
                let signal_handle = FstSignalHandle::from_index(idx);
                match signal_length {
                    0 => {} // ignore since variable-length records have no initial value
                    len => {
                        if !self.meta.signals[idx].is_real() {
                            let value = read_bytes(&mut byte_reader, len as usize)?;
                            (self.callback)(
                                start_time,
                                signal_handle,
                                FstSignalValue::String(std::str::from_utf8(&value)?),
                            );
                        } else {
                            let value = read_f64(&mut byte_reader, self.meta.float_endian)?;
                            (self.callback)(start_time, signal_handle, FstSignalValue::Real(value));
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
                match shval.cmp(&0) {
                    Ordering::Greater => {
                        value += shval;
                        if !chain_table.is_empty() {
                            let len = (value - chain_table[prev_idx]) as u32;
                            chain_table_lengths[prev_idx] = len;
                        }
                        prev_idx = idx;
                        chain_table.push(value);
                    }
                    Ordering::Less => {
                        chain_table.push(0);
                        prev_alias = shval as u32;
                        chain_table_lengths[idx] = prev_alias;
                    }
                    Ordering::Equal => {
                        chain_table.push(0);
                        chain_table_lengths[idx] = prev_alias;
                    }
                }
            } else {
                let (value, _) = read_variant_u32(&mut chain_bytes)?;
                let zeros = value >> 1;
                push_zeros(&mut chain_table, zeros);
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
                chain_table.push(0); // alias
                let (len, _) = read_variant_u32(&mut chain_bytes)?;
                chain_table_lengths[idx] = (-(len as i64)) as u32;
            } else if (raw_val & 1) == 1 {
                value += (raw_val as i64) >> 1;
                if idx > 0 {
                    let len = (value - chain_table[prev_idx]) as u32;
                    chain_table_lengths[prev_idx] = len;
                }
                chain_table.push(value);
                prev_idx = idx; // only take non-alias signals into account
            } else {
                let zeros = raw_val >> 1;
                push_zeros(&mut chain_table, zeros);
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
            // was there a signal change?
            if *entry != 0 {
                // is the signal supposed to be included?
                if self.filter.signals[signal_idx] {
                    // read all signal values
                    self.input
                        .seek(SeekFrom::Start((vc_start as i64 + entry) as u64))?;
                    let mut bytes = read_packed_signal_values(&mut self.input, *length, packtpe)?;

                    // read first time delta
                    let len = self.meta.signals[signal_idx].len();
                    let tdelta = if len == 1 {
                        read_one_bit_signal_time_delta(&bytes, 0)?
                    } else {
                        read_multi_bit_signal_time_delta(&bytes, 0)?
                    };

                    // remember where we stored the signal data and how long it is
                    head_pointer.push(mu.len() as u32);
                    length_remaining.push(bytes.len() as u32);
                    mu.append(&mut bytes);

                    // remember at what time step we will read this signal
                    scatter_pointer[signal_idx] = tc_head[tdelta];
                    tc_head[tdelta] = signal_idx as u32 + 1; // index to handle
                }
            }
            // if there was no real value added, we add dummy values to ensure that we can
            // index the Vec with the signal ID
            if head_pointer.len() == signal_idx {
                head_pointer.push(1234);
                length_remaining.push(1234);
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
                let signal_id = (tc_head[time_id] - 1) as usize; // convert handle to index
                let mut mu_slice = &mu.as_slice()[head_pointer[signal_id] as usize..];
                let (vli, skiplen) = read_variant_u32(&mut mu_slice)?;
                let signal_len = self.meta.signals[signal_id].len();
                let signal_handle = FstSignalHandle::from_index(signal_id);
                let len = match signal_len {
                    1 => {
                        let value = one_bit_signal_value_to_char(vli);
                        let value_buf = [value];
                        (self.callback)(
                            *time,
                            signal_handle,
                            FstSignalValue::String(std::str::from_utf8(&value_buf)?),
                        );
                        0 // no additional bytes consumed
                    }
                    0 => {
                        let (len, skiplen2) = read_variant_u32(&mut mu_slice)?;
                        let value = read_bytes(&mut mu_slice, len as usize)?;
                        (self.callback)(
                            *time,
                            signal_handle,
                            FstSignalValue::String(std::str::from_utf8(&value)?),
                        );
                        len + skiplen2
                    }
                    len => {
                        let signal_len = len as usize;
                        if !self.meta.signals[signal_id].is_real() {
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
                            (self.callback)(
                                *time,
                                signal_handle,
                                FstSignalValue::String(std::str::from_utf8(&value)?),
                            );
                            len
                        } else {
                            assert_eq!(vli & 1, 1, "TODO: implement support for rare packed case");
                            let value = read_f64(&mut mu_slice, self.meta.float_endian)?;
                            (self.callback)(*time, signal_handle, FstSignalValue::Real(value));
                            8
                        }
                    }
                };

                // update pointers
                let total_skiplen = skiplen + len;
                // advance "slice" for signal values
                head_pointer[signal_id] += total_skiplen;
                length_remaining[signal_id] -= total_skiplen;
                // find the next signal to read in this time step
                tc_head[time_id] = scatter_pointer[signal_id];
                // invalidate pointer
                scatter_pointer[signal_id] = 0;

                // is there more data for this signal in the current block?
                if length_remaining[signal_id] > 0 {
                    let tdelta = if signal_len == 1 {
                        read_one_bit_signal_time_delta(&mu, head_pointer[signal_id])?
                    } else {
                        read_multi_bit_signal_time_delta(&mu, head_pointer[signal_id])?
                    };

                    // point to the next time step
                    scatter_pointer[signal_id] = tc_head[time_id + tdelta];
                    tc_head[time_id + tdelta] = (signal_id + 1) as u32; // store handle
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
