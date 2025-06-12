// Copyright 2023 The Regents of the University of California
// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::io::*;
use crate::types::*;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};

trait BufReadSeek: BufRead + Seek {}
impl<T> BufReadSeek for T where T: BufRead + Seek {}

/// Reads in a FST file.
pub struct FstReader<R: BufRead + Seek> {
    input: InputVariant<R>,
    meta: MetaData,
}

enum InputVariant<R: BufRead + Seek> {
    Original(R),
    Incomplete(R, Box<dyn BufReadSeek + Sync + Send>),
    UncompressedInMem(std::io::Cursor<Vec<u8>>),
    IncompleteUncompressedInMem(std::io::Cursor<Vec<u8>>, Box<dyn BufReadSeek + Sync + Send>),
}

/// Filter the changes by time and/or signals
///
/// The time filter is inclusive, i.e. it includes all changes in `start..=end`.
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
    /// the exponent of the timescale; timescale will be 10^(exponent) seconds
    pub timescale_exponent: i8,
}

impl<R: BufRead + Seek> FstReader<R> {
    /// Reads in the FST file meta-data.
    pub fn open(input: R) -> Result<Self> {
        Self::open_internal(input, false)
    }

    pub fn open_and_read_time_table(input: R) -> Result<Self> {
        Self::open_internal(input, true)
    }

    fn open_internal(mut input: R, read_time_table: bool) -> Result<Self> {
        let uncompressed_input = uncompress_gzip_wrapper(&mut input)?;
        match uncompressed_input {
            UncompressGzipWrapper::None => {
                let mut header_reader = HeaderReader::new(input);
                header_reader.read(read_time_table)?;
                let (input, meta) = header_reader.into_input_and_meta_data().unwrap();
                Ok(FstReader {
                    input: InputVariant::Original(input),
                    meta,
                })
            }
            UncompressGzipWrapper::InMemory(uc) => {
                let mut header_reader = HeaderReader::new(uc);
                header_reader.read(read_time_table)?;
                let (uc2, meta) = header_reader.into_input_and_meta_data().unwrap();
                Ok(FstReader {
                    input: InputVariant::UncompressedInMem(uc2),
                    meta,
                })
            }
        }
    }

    /// Reads in the meta-data of an incomplete FST file with external `.hier` file.
    ///
    /// When the creation of an FST file using `fstlib` is interrupted, some information
    /// such as a geometry block or hierarchy block may be missing (as indicated by
    /// [`ReaderError::MissingGeometry`] and [`ReaderError::MissingHierarchy`]).
    ///
    /// This function tries to reconstruct these missing blocks from an external `.hier`
    /// file, which is commonly generated while outputting FST files.
    pub fn open_incomplete<H: BufRead + Seek + Sync + Send + 'static>(
        input: R,
        hierarchy: H,
    ) -> Result<Self> {
        Self::open_incomplete_internal(input, hierarchy, false)
    }

    pub fn open_incomplete_and_read_time_table<H: BufRead + Seek + Sync + Send + 'static>(
        input: R,
        hierarchy: H,
    ) -> Result<Self> {
        Self::open_incomplete_internal(input, hierarchy, true)
    }

    fn open_incomplete_internal<H: BufRead + Seek + Sync + Send + 'static>(
        mut input: R,
        mut hierarchy: H,
        read_time_table: bool,
    ) -> Result<Self> {
        let uncompressed_input = uncompress_gzip_wrapper(&mut input)?;
        match uncompressed_input {
            UncompressGzipWrapper::None => {
                let (input, meta) = Self::open_incomplete_internal_uncompressed(
                    input,
                    &mut hierarchy,
                    read_time_table,
                )?;
                Ok(FstReader {
                    input: InputVariant::Incomplete(input, Box::new(hierarchy)),
                    meta,
                })
            }
            UncompressGzipWrapper::InMemory(uc) => {
                let (uc2, meta) = Self::open_incomplete_internal_uncompressed(
                    uc,
                    &mut hierarchy,
                    read_time_table,
                )?;
                Ok(FstReader {
                    input: InputVariant::IncompleteUncompressedInMem(uc2, Box::new(hierarchy)),
                    meta,
                })
            }
        }
    }

    fn open_incomplete_internal_uncompressed<I: BufRead + Seek, H: BufRead + Seek>(
        input: I,
        hierarchy: &mut H,
        read_time_table: bool,
    ) -> Result<(I, MetaData)> {
        let mut header_reader = HeaderReader::new(input);
        match header_reader.read(read_time_table) {
            Ok(_) => {}
            Err(ReaderError::MissingGeometry() | ReaderError::MissingHierarchy()) => {
                header_reader
                    .hierarchy
                    .get_or_insert((HierarchyCompression::Uncompressed, 0));
                header_reader.reconstruct_geometry(hierarchy)?;
            }
            Err(e) => return Err(e),
        };
        Ok(header_reader.into_input_and_meta_data().unwrap())
    }

    pub fn get_header(&self) -> FstHeader {
        FstHeader {
            start_time: self.meta.header.start_time,
            end_time: self.meta.header.end_time,
            var_count: self.meta.header.var_count,
            max_handle: self.meta.header.max_var_id_code,
            version: self.meta.header.version.clone(),
            date: self.meta.header.date.clone(),
            timescale_exponent: self.meta.header.timescale_exponent,
        }
    }

    pub fn get_time_table(&self) -> Option<&[u64]> {
        match &self.meta.time_table {
            Some(table) => Some(table),
            None => None,
        }
    }

    /// Reads the hierarchy and calls callback for every item.
    pub fn read_hierarchy(&mut self, callback: impl FnMut(FstHierarchyEntry)) -> Result<()> {
        match &mut self.input {
            InputVariant::Original(input) => read_hierarchy(input, &self.meta, callback),
            InputVariant::Incomplete(_, input) => read_hierarchy(input, &self.meta, callback),
            InputVariant::UncompressedInMem(input) => read_hierarchy(input, &self.meta, callback),
            InputVariant::IncompleteUncompressedInMem(_, input) => {
                read_hierarchy(input, &self.meta, callback)
            }
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
            InputVariant::Incomplete(input, _) => {
                read_signals(input, &self.meta, &data_filter, callback)
            }
            InputVariant::UncompressedInMem(input) => {
                read_signals(input, &self.meta, &data_filter, callback)
            }
            InputVariant::IncompleteUncompressedInMem(input, _) => {
                read_signals(input, &self.meta, &data_filter, callback)
            }
        }
    }
}

pub enum FstSignalValue<'a> {
    String(&'a [u8]),
    Real(f64),
}

/// Quickly scans an input to see if it could be a FST file.
pub fn is_fst_file(input: &mut (impl Read + Seek)) -> bool {
    let is_fst = matches!(internal_check_fst_file(input), Ok(true));
    // try to reset input
    let _ = input.seek(SeekFrom::Start(0));
    is_fst
}

/// Returns an error or false if not an fst. Returns Ok(true) only if we think it is an fst.
fn internal_check_fst_file(input: &mut (impl Read + Seek)) -> Result<bool> {
    let mut seen_header = false;

    // try to iterate over all blocks
    loop {
        let block_tpe = match read_block_tpe(input) {
            Err(ReaderError::Io(_)) => {
                break;
            }
            Err(other) => return Err(other),
            Ok(tpe) => tpe,
        };
        let section_length = read_u64(input)?;
        match block_tpe {
            BlockType::GZipWrapper => return Ok(true),
            BlockType::Header => {
                seen_header = true;
            }
            BlockType::Skip if section_length == 0 => {
                break;
            }
            _ => {}
        }
        input.seek(SeekFrom::Current((section_length as i64) - 8))?;
    }
    Ok(seen_header)
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

enum UncompressGzipWrapper {
    None,
    // TempFile(BufReader<std::fs::File>),
    InMemory(std::io::Cursor<Vec<u8>>),
}

/// Checks to see if the whole file is compressed in which case it is decompressed
/// to a temp file which is returned.
fn uncompress_gzip_wrapper(input: &mut (impl Read + Seek)) -> Result<UncompressGzipWrapper> {
    let block_tpe = read_block_tpe(input)?;
    if block_tpe != BlockType::GZipWrapper {
        // no gzip wrapper
        input.seek(SeekFrom::Start(0))?;
        Ok(UncompressGzipWrapper::None)
    } else {
        // uncompress
        let section_length = read_u64(input)?;
        let uncompress_length = read_u64(input)? as usize;
        if section_length == 0 {
            return Err(ReaderError::NotFinishedCompressing());
        }

        // TODO: add back the ability to uncompress to a temporary file without adding a dependency
        // we always decompress into memory
        let mut target = vec![];
        decompress_gz_in_chunks(input, uncompress_length, &mut target)?;
        let new_input = std::io::Cursor::new(target);
        Ok(UncompressGzipWrapper::InMemory(new_input))
    }
}

fn decompress_gz_in_chunks(
    input: &mut (impl Read + Seek),
    mut remaining: usize,
    target: &mut impl Write,
) -> Result<()> {
    read_gzip_header(input)?;
    let mut buf_in = vec![0u8; 32768 / 2]; // FST_GZIO_LEN
    let mut buf_out = vec![0u8; 32768 * 2]; // FST_GZIO_LEN

    let mut state = miniz_oxide::inflate::stream::InflateState::new(miniz_oxide::DataFormat::Raw);
    let mut buf_in_remaining = 0;
    while remaining > 0 {
        // load more bytes into the input buffer
        buf_in_remaining += input.read(&mut buf_in[buf_in_remaining..])?;
        debug_assert!(
            buf_in_remaining > 0,
            "ran out of input data while gzip decompressing"
        );

        // decompress them
        let res = miniz_oxide::inflate::stream::inflate(
            &mut state,
            &buf_in[0..buf_in_remaining],
            buf_out.as_mut_slice(),
            miniz_oxide::MZFlush::None,
        );

        println!("{res:?}\nremaining={remaining}");

        match res.status {
            Ok(status) => {
                // move bytes that were not consumed to the start of the buffer and update the length
                buf_in.copy_within(res.bytes_consumed..buf_in_remaining, 0);
                buf_in_remaining -= res.bytes_consumed;

                // write decompressed output
                let out_bytes = std::cmp::min(res.bytes_written, remaining);
                remaining -= out_bytes;
                target.write_all(&buf_out[..out_bytes])?;

                match status {
                    miniz_oxide::MZStatus::Ok => {
                        // nothing to do
                    }
                    miniz_oxide::MZStatus::StreamEnd => {
                        debug_assert_eq!(remaining, 0);
                        return Ok(());
                    }
                    miniz_oxide::MZStatus::NeedDict => {
                        todo!("hande NeedDict status");
                    }
                }
            }
            Err(err) => {
                return Err(ReaderError::GZipBody(format!("{err:?}")));
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
struct MetaData {
    header: Header,
    signals: Vec<SignalInfo>,
    #[allow(dead_code)]
    blackouts: Vec<BlackoutData>,
    data_sections: Vec<DataSectionInfo>,
    float_endian: FloatingPointEndian,
    hierarchy_compression: HierarchyCompression,
    hierarchy_offset: u64,
    time_table: Option<Vec<u64>>,
}

pub type Result<T> = std::result::Result<T, ReaderError>;

struct HeaderReader<R: Read + Seek> {
    input: R,
    header: Option<Header>,
    signals: Option<Vec<SignalInfo>>,
    blackouts: Option<Vec<BlackoutData>>,
    data_sections: Vec<DataSectionInfo>,
    float_endian: FloatingPointEndian,
    hierarchy: Option<(HierarchyCompression, u64)>,
    time_table: Option<Vec<u64>>,
}

impl<R: Read + Seek> HeaderReader<R> {
    fn new(input: R) -> Self {
        HeaderReader {
            input,
            header: None,
            signals: None,
            blackouts: None,
            data_sections: Vec::default(),
            float_endian: FloatingPointEndian::Little,
            hierarchy: None,
            time_table: None,
        }
    }

    fn read_data(&mut self, tpe: &BlockType) -> Result<()> {
        let file_offset = self.input.stream_position()?;
        // this is the data section
        let section_length = read_u64(&mut self.input)?;
        let start_time = read_u64(&mut self.input)?;
        let end_time = read_u64(&mut self.input)?;
        let mem_required_for_traversal = read_u64(&mut self.input)?;

        // optional: read the time table
        if let Some(table) = &mut self.time_table {
            let (_, mut time_chain) =
                read_time_table(&mut self.input, file_offset, section_length)?;
            // in the first section, we might need to include the start time
            let is_first_section = table.is_empty();
            if is_first_section && time_chain[0] > start_time {
                table.push(start_time);
            }
            table.append(&mut time_chain);
            self.input.seek(SeekFrom::Start(file_offset + 4 * 8))?;
        }
        // go to the end of the section
        self.skip(section_length, 4 * 8)?;
        let kind = DataSectionKind::from_block_type(tpe).unwrap();
        let info = DataSectionInfo {
            file_offset,
            start_time,
            end_time,
            kind,
            mem_required_for_traversal,
        };

        // incomplete fst files may have start_time and end_time set to 0,
        // in which case we can infer it from the data
        if let Some(Header {
            start_time: header_start,
            end_time: header_end,
            ..
        }) = self.header.as_mut()
        {
            if *header_start == 0 && *header_end == 0 {
                *header_end = end_time;
                if self.data_sections.is_empty() {
                    *header_start = start_time;
                }
            }
        }

        self.data_sections.push(info);
        Ok(())
    }

    fn skip(&mut self, section_length: u64, already_read: i64) -> Result<u64> {
        Ok(self
            .input
            .seek(SeekFrom::Current((section_length as i64) - already_read))?)
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

    // The geometry block contains the types and lengths of variables (see [`SignalInfo`]).
    // In case this block is missing from the FST file, it can be reconstructed from the hierarchy.
    fn reconstruct_geometry(&mut self, hierarchy: &mut (impl BufRead + Seek)) -> Result<()> {
        hierarchy.seek(SeekFrom::Start(0))?;
        let bytes = read_hierarchy_bytes(hierarchy, HierarchyCompression::Uncompressed)?;
        let mut input = bytes.as_slice();
        let mut handle_count = 0u32;
        let mut signals: Vec<SignalInfo> = Vec::new();
        while let Some(entry) = read_hierarchy_entry(&mut input, &mut handle_count)? {
            match entry {
                FstHierarchyEntry::Var {
                    length, is_alias, ..
                } if !is_alias => {
                    signals.push(SignalInfo::from_file_format(length));
                }
                _ => {}
            }
        }
        self.signals = Some(signals);
        Ok(())
    }

    fn read(&mut self, read_time_table: bool) -> Result<()> {
        if read_time_table {
            self.time_table = Some(Vec::new());
        }
        loop {
            let block_tpe = match read_block_tpe(&mut self.input) {
                Err(ReaderError::Io(_)) => {
                    break;
                }
                Err(other) => return Err(other),
                Ok(tpe) => tpe,
            };

            dbg!(&block_tpe);

            match block_tpe {
                BlockType::Header => {
                    let (header, endian) = read_header(&mut self.input)?;
                    self.header = Some(header);
                    self.float_endian = endian;
                }
                BlockType::VcData => self.read_data(&block_tpe)?,
                BlockType::VcDataDynamicAlias => self.read_data(&block_tpe)?,
                BlockType::VcDataDynamicAlias2 => self.read_data(&block_tpe)?,
                BlockType::Blackout => {
                    self.blackouts = Some(read_blackout(&mut self.input)?);
                }
                BlockType::Geometry => {
                    self.signals = Some(read_geometry(&mut self.input)?);
                }
                BlockType::Hierarchy => self.read_hierarchy(HierarchyCompression::ZLib)?,
                BlockType::HierarchyLZ4 => self.read_hierarchy(HierarchyCompression::Lz4)?,
                BlockType::HierarchyLZ4Duo => self.read_hierarchy(HierarchyCompression::Lz4Duo)?,
                BlockType::GZipWrapper => panic!("GZip Wrapper should have been handled earlier!"),
                BlockType::Skip => {
                    let section_length = read_u64(&mut self.input)?;
                    if section_length == 0 {
                        break;
                    }
                    self.skip(section_length, 8)?;
                }
            };
        }

        if self.signals.is_none() {
            return Err(ReaderError::MissingGeometry());
        }

        if self.hierarchy.is_none() {
            return Err(ReaderError::MissingHierarchy());
        }

        Ok(())
    }

    fn into_input_and_meta_data(mut self) -> Result<(R, MetaData)> {
        self.input.seek(SeekFrom::Start(0))?;
        let meta = MetaData {
            header: self.header.unwrap(),
            signals: self.signals.unwrap(),
            blackouts: self.blackouts.unwrap_or_default(),
            data_sections: self.data_sections,
            float_endian: self.float_endian,
            hierarchy_compression: self.hierarchy.unwrap().0,
            hierarchy_offset: self.hierarchy.unwrap().1,
            time_table: self.time_table,
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

impl<R: Read + Seek, F: FnMut(u64, FstSignalHandle, FstSignalValue)> DataReader<'_, R, F> {
    fn read_value_changes(
        &mut self,
        section_kind: DataSectionKind,
        section_start: u64,
        section_length: u64,
        time_section_length: u64,
        time_table: &[u64],
    ) -> Result<()> {
        let (max_handle, _) = read_variant_u64(&mut self.input)?;
        let vc_start = self.input.stream_position()?;
        let packtpe = ValueChangePackType::from_u8(read_u8(&mut self.input)?);
        // the chain length is right in front of the time section
        let chain_len_offset = section_start + section_length - time_section_length - 8;
        let signal_offsets = read_signal_locs(
            &mut self.input,
            chain_len_offset,
            section_kind,
            max_handle,
            vc_start,
        )?;

        // read data and create a bunch of pointers
        let mut mu: Vec<u8> = Vec::new();
        let mut head_pointer = vec![0u32; max_handle as usize];
        let mut length_remaining = vec![0u32; max_handle as usize];
        let mut scatter_pointer = vec![0u32; max_handle as usize];
        let mut tc_head = vec![0u32; std::cmp::max(1, time_table.len())];

        for entry in signal_offsets.iter() {
            // is the signal supposed to be included?
            if self.filter.signals.is_set(entry.signal_idx) {
                // read all signal values
                self.input.seek(SeekFrom::Start(vc_start + entry.offset))?;
                let mut bytes =
                    read_packed_signal_value_bytes(&mut self.input, entry.len, packtpe)?;

                // read first time delta
                let len = self.meta.signals[entry.signal_idx].len();
                let tdelta = if len == 1 {
                    read_one_bit_signal_time_delta(&bytes, 0)?
                } else {
                    read_multi_bit_signal_time_delta(&bytes, 0)?
                };

                // remember where we stored the signal data and how long it is
                head_pointer[entry.signal_idx] = mu.len() as u32;
                length_remaining[entry.signal_idx] = bytes.len() as u32;
                mu.append(&mut bytes);

                // remember at what time step we will read this signal
                scatter_pointer[entry.signal_idx] = tc_head[tdelta];
                tc_head[tdelta] = entry.signal_idx as u32 + 1; // index to handle
            }
        }

        let mut buffer = Vec::new();

        for (time_id, time) in time_table.iter().enumerate() {
            // while we cannot ignore signal changes before the start of the window
            // (since the signal might retain values for multiple cycles),
            // signal changes after our window are completely useless
            if *time > self.filter.end {
                break;
            }

            let eof_error = || {
                ReaderError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected eof",
                ))
            };

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
                        (self.callback)(*time, signal_handle, FstSignalValue::String(&value_buf));
                        0 // no additional bytes consumed
                    }
                    0 => {
                        let (len, skiplen2) = read_variant_u32(&mut mu_slice)?;
                        let value = mu_slice.get(..len as usize).ok_or_else(eof_error)?;
                        (self.callback)(*time, signal_handle, FstSignalValue::String(value));
                        len + skiplen2
                    }
                    len => {
                        let signal_len = len as usize;
                        if !self.meta.signals[signal_id].is_real() {
                            let (value, len) = if (vli & 1) == 0 {
                                // if bit0 is zero -> 2-state
                                let read_len = signal_len.div_ceil(8);
                                let bytes = mu_slice.get(..read_len).ok_or_else(eof_error)?;
                                multi_bit_digital_signal_to_chars(bytes, signal_len, &mut buffer);
                                (buffer.as_slice(), read_len as u32)
                            } else {
                                let value = mu_slice.get(..signal_len).ok_or_else(eof_error)?;
                                (value, len)
                            };
                            (self.callback)(*time, signal_handle, FstSignalValue::String(value));
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

            // read the time table
            let (time_section_length, time_table) =
                read_time_table(&mut self.input, section.file_offset, section_length)?;

            // only read frame if this is the first section and there is no other data for
            // the start time
            if is_first_section && (time_table.is_empty() || time_table[0] > start_time) {
                read_frame(
                    &mut self.input,
                    section.file_offset,
                    section_length,
                    &self.meta.signals,
                    &self.filter.signals,
                    self.meta.float_endian,
                    start_time,
                    self.callback,
                )?;
            } else {
                skip_frame(&mut self.input, section.file_offset)?;
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
