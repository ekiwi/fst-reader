use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::io::{Read, Seek, SeekFrom};

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
enum FileType {
    Verilog = 0,
    Vhdl = 1,
    VerilogVhdl = 2,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
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
enum ScopeType {
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
#[derive(Debug, TryFromPrimitive, PartialEq)]
enum VarType {
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
enum VarDirection {
    Implicit = 0,
    Input = 1,
    Output = 2,
    InOut = 3,
    Buffer = 4,
    Linkage = 5,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
enum AttributeType {
    Misc = 0,
    Array = 1,
    Enum = 2,
    Pack = 3,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
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
    types: Vec<VarType>,
}

#[derive(Debug)]
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
}

#[derive(Debug)]
struct Handle(u32);
#[derive(Debug)]
struct EnumHandle(u32);

#[derive(Debug)]
enum HierarchyEntry {
    Scope {
        tpe: ScopeType,
        name: String,
        component: String,
    },
    UpScope,
    Var {
        tpe: VarType,
        direction: VarDirection,
        name: String,
        length: u32,
        handle: Handle,
        is_alias: bool,
    },
    AttributeBegin {
        name: String,
        // TODO
    },
    AttributeEnd,
}

#[derive(Debug)]
enum HierarchyCompression {
    ZLib,
    Lz4,
    Lz4Duo,
}

#[derive(Debug)]
enum ReaderErrorKind {
    IO(std::io::Error),
    FromPrimitive(),
    StringParse(std::str::Utf8Error),
    StringParse2(std::string::FromUtf8Error),
    ParseVariant(),
    Decompress(lz4_flex::block::DecompressError),
}
#[derive(Debug)]
struct ReaderError {
    kind: ReaderErrorKind,
}

impl From<std::io::Error> for ReaderError {
    fn from(value: std::io::Error) -> Self {
        let kind = ReaderErrorKind::IO(value);
        ReaderError { kind }
    }
}

impl<Enum: TryFromPrimitive> From<TryFromPrimitiveError<Enum>> for ReaderError {
    fn from(value: TryFromPrimitiveError<Enum>) -> Self {
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
        let kind = ReaderErrorKind::Decompress(value);
        ReaderError { kind }
    }
}

type Result<T> = std::result::Result<T, ReaderError>;

struct HeaderReader<R: Read + Seek> {
    input: FstReader<R>,
    header: Option<Header>,
    signals: Option<Signals>,
    data_sections: Vec<DataSectionInfo>,
}

struct HierarchyReader<R: Read> {
    input: FstReader<R>,
}

#[inline]
fn read_variant_32(input: &mut impl Read) -> Result<u32> {
    let mut byte = [0u8; 1];
    let mut res = 0u32;
    for ii in 0..5 {
        // 32bit / 7bit = ~4.6
        input.read_exact(&mut byte)?;
        let value = (byte[0] as u32) & 0x7f;
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
fn read_variant_64(input: &mut impl Read) -> Result<u64> {
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

const HIERARCHY_NAME_MAX_SIZE: usize = 512;
const HIERARCHY_ATTRIBUTE_MAX_SIZE: usize = 65536 + 4096;

struct FstReader<R: Read> {
    input: R,
    buf: [u8; 128], // used for reading
}

impl<R: Read> FstReader<R> {
    fn new(input: R) -> Self {
        FstReader {
            input,
            buf: [0u8; 128],
        }
    }

    #[inline]
    fn read_variant_32(&mut self) -> Result<u32> {
        read_variant_32(&mut self.input)
    }

    fn read_c_str(&mut self, max_len: usize) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::with_capacity(32);
        loop {
            let byte = self.read_u8()?;
            if byte == 0 {
                break;
            } else {
                bytes.push(byte);
            }
        }
        Ok(String::from_utf8(bytes)?)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.input.read_exact(&mut self.buf[..8])?;
        Ok(u64::from_be_bytes(self.buf[..8].try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.input.read_exact(&mut self.buf[..8])?;
        Ok(f64::from_le_bytes(self.buf[..8].try_into().unwrap()))
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.input.read_exact(&mut self.buf[..1])?;
        Ok(self.buf[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        self.input.read_exact(&mut self.buf[..1])?;
        Ok(i8::from_be_bytes(self.buf[..1].try_into().unwrap()))
    }

    #[inline] // inline to specialize on length
    fn read_string(&mut self, len: usize) -> Result<String> {
        assert!(len <= 128);
        self.input.read_exact(&mut self.buf[..len])?;
        let zero_index = self.buf.iter().position(|b| *b == 0u8).unwrap_or(len - 1);
        Ok((std::str::from_utf8(&self.buf[..(zero_index + 1)])?).to_string())
    }

    fn read_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        self.input.by_ref().take(len as u64).read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn read_block_tpe(&mut self) -> Result<BlockType> {
        Ok(BlockType::try_from(self.read_u8()?)?)
    }
}

impl<R: Read + Seek> FstReader<R> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.input.seek(pos)
    }
}

impl<R: Read> HierarchyReader<R> {
    fn new(input: R) -> Self {
        HierarchyReader {
            input: FstReader::new(input),
        }
    }

    fn read_all_entries(&mut self) -> Result<()> {
        loop {
            let entry = match self.read_entry()? {
                Some(e) => e,
                None => return Ok(()),
            };
            println!("{entry:?}");
        }
    }

    fn read_entry(&mut self) -> Result<Option<HierarchyEntry>> {
        let entry_tpe = match self.input.read_u8() {
            Ok(tpe) => tpe,
            Err(e) => return Ok(None),
        };
        let entry = match entry_tpe {
            254 => {
                // VcdScope (ScopeType)
                let tpe = ScopeType::try_from_primitive(self.input.read_u8()?)?;
                let name = self.input.read_c_str(HIERARCHY_NAME_MAX_SIZE)?;
                let component = self.input.read_c_str(HIERARCHY_NAME_MAX_SIZE)?;
                HierarchyEntry::Scope {
                    tpe,
                    name,
                    component,
                }
            }
            0..=29 => {
                // VcdEvent ... SvShortReal (VariableType)
                let tpe = VarType::try_from_primitive(entry_tpe)?;
                let direction = VarDirection::try_from_primitive(self.input.read_u8()?)?;
                let name = self.input.read_c_str(HIERARCHY_NAME_MAX_SIZE)?;
                let raw_length = self.input.read_variant_32()?;
                let length = if tpe == VarType::Port {
                    // remove delimiting spaces and adjust signal size
                    (raw_length - 2) / 3
                } else {
                    raw_length
                };
                let alias = self.input.read_variant_32()?;
                let (is_alias, handle) = if alias == 0 {
                    (false, Handle(0)) // TODO: create unique handle by counting!
                } else {
                    (true, Handle(alias))
                };
                HierarchyEntry::Var {
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
                HierarchyEntry::UpScope
            }
            252 => {
                // GenAttributeBegin (ScopeType)
                todo!("Deal with Attribute Begin entry!")
            }
            253 => {
                // GenAttributeEnd (ScopeType)
                HierarchyEntry::AttributeEnd
            }

            other => todo!("Deal with hierarchy entry of type: {other}"),
        };

        Ok(Some(entry))
    }
}

impl<R: Read + Seek> HeaderReader<R> {
    fn new(input: R) -> Self {
        HeaderReader {
            input: FstReader::new(input),
            header: None,
            signals: None,
            data_sections: Vec::default(),
        }
    }

    fn header_incomplete(&self) -> bool {
        match &self.header {
            None => true,
            Some(h) => h.start_time == 0 && h.end_time == 0,
        }
    }

    fn read_header(&mut self) -> Result<()> {
        let section_length = self.input.read_u64()?;
        assert_eq!(section_length, 329);
        let start_time = self.input.read_u64()?;
        let end_time = self.input.read_u64()?;
        let endian_test = self.input.read_f64()?;
        assert_eq!(endian_test, DOUBLE_ENDIAN_TEST);
        let memory_used_by_writer = self.input.read_u64()?;
        let scope_count = self.input.read_u64()?;
        let var_count = self.input.read_u64()?;
        let max_var_id_code = self.input.read_u64()?;
        let vc_section_count = self.input.read_u64()?;
        let timescale_exponent = self.input.read_i8()?;
        let version = self.input.read_string(128)?;
        // this size was reduced compared to what is documented in block_format.txt
        let date = self.input.read_string(119)?;
        let file_type = FileType::try_from(self.input.read_u8()?)?;
        let time_zero = self.input.read_u64()?;

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
        let file_offset = self.input.input.stream_position()?;
        // this is the data section
        let section_length = self.input.read_u64()?;
        let start_time = self.input.read_u64()?;
        let end_time = self.input.read_u64()?;
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
        let section_length = self.input.read_u64()?;
        // skip this section if the header is not complete
        if self.header_incomplete() {
            self.skip(section_length, 8)?;
            return Ok(());
        }

        let uncompressed_length = self.input.read_u64()?;
        let max_handle = self.input.read_u64()?;
        let compressed_length = section_length - 3 * 8;

        let bytes = if compressed_length == uncompressed_length {
            self.input.read_bytes(uncompressed_length as usize)?
        } else {
            todo!("add support for decompression")
        };

        println!("max_handle = {max_handle}");
        let mut longest_signal_value_len = 32;
        let mut lengths: Vec<u32> = Vec::with_capacity(max_handle as usize);
        let mut types: Vec<VarType> = Vec::with_capacity(max_handle as usize);
        let mut byte_reader: &[u8] = &bytes;

        for ii in 0..max_handle {
            let value = read_variant_32(&mut byte_reader)?;
            let (length, tpe) = if value == 0 {
                (8, VarType::Real)
            } else {
                let length = if value != 0xFFFFFFFFu32 { value } else { 0 };
                if length > longest_signal_value_len {
                    // TODO: is this code ever run?
                    longest_signal_value_len = length;
                }
                (length, VarType::Wire)
            };
            println!("{ii} {length} {tpe:?}");
            lengths.push(length);
            types.push(tpe);
        }
        self.signals = Some(Signals { lengths, types });

        Ok(())
    }

    fn read_hierarchy_bytes(&mut self, compression: HierarchyCompression) -> Result<Vec<u8>> {
        // Note: the GTKWave implementation skips decoding the hierarchy block initially
        // it also first re-creates a history file and then read from it.
        // Here we only recreate things in memory.
        // This function is similar to fstReaderRecreateHierFile

        let section_length = self.input.read_u64()? as usize;
        let uncompressed_length = self.input.read_u64()? as usize;
        let compressed_length = section_length - 2 * 8;

        let bytes = match compression {
            HierarchyCompression::ZLib => todo!("ZLib compression is currently not supported!"),
            HierarchyCompression::Lz4 => {
                let compressed = self.input.read_bytes(compressed_length)?;
                let uncompressed = lz4_flex::decompress(&compressed, uncompressed_length as usize)?;
                uncompressed
            }
            HierarchyCompression::Lz4Duo => todo!("Implement LZ4 Duo!"),
        };
        assert_eq!(bytes.len(), uncompressed_length);
        Ok(bytes)
    }

    fn read_hierarchy(&mut self, compression: HierarchyCompression) -> Result<()> {
        // similar to fstReaderIterateHier
        let bytes = self.read_hierarchy_bytes(compression)?;
        let mut reader = HierarchyReader::new(bytes.as_slice());
        reader.read_all_entries()?;

        Ok(())
    }

    fn read(&mut self) -> Result<()> {
        loop {
            let block_tpe = match self.input.read_block_tpe() {
                Err(_) => break,
                Ok(tpe) => tpe,
            };
            println!("{block_tpe:?}");
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
        };
        Ok((self.input.input, meta))
    }
}

struct DataReader<R: Read + Seek> {
    input: FstReader<R>,
    meta: MetaData,
}

#[derive(Debug)]
enum DataSectionKind {
    Standard,
    DynamicAlias,
    DynamicAlias2,
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

impl<R: Read + Seek> DataReader<R> {
    fn new(input: R, meta: MetaData) -> Self {
        DataReader {
            input: FstReader::new(input),
            meta,
        }
    }

    fn read_time_block(&mut self, section_start: u64, section_length: u64) -> Result<Vec<u64>> {
        // the time block meta data is in the last 24 bytes at the end of the section
        self.input
            .seek(SeekFrom::Start(section_start + section_length - 3 * 8))?;
        let uncompressed_length = self.input.read_u64()?;
        let compressed_length = self.input.read_u64()?;
        let number_of_items = self.input.read_u64()?;
        assert!(compressed_length <= section_length);

        // now that we know how long the block actually is, we can go back to it
        self.input
            .seek(SeekFrom::Current(-(3 * 8) - (compressed_length as i64)))?;
        let bytes = if compressed_length == uncompressed_length {
            self.input.read_bytes(uncompressed_length as usize)?
        } else {
            todo!("add support for decompression")
        };
        let mut byte_reader: &[u8] = &bytes;
        let mut time_table: Vec<u64> = Vec::with_capacity(number_of_items as usize);
        let mut time_val: u64 = 0; // running time counter

        for _ in 0..number_of_items {
            let value = read_variant_64(&mut byte_reader)?;
            time_val += value;
            time_table.push(time_val);
        }

        Ok(time_table)
    }

    fn read(&mut self) -> Result<()> {
        for section in self.meta.data_sections.iter() {
            // skip to section
            self.input.seek(SeekFrom::Start(section.file_offset))?;
            let section_length = self.input.read_u64()?;

            // verify meta-data
            let start_time = self.input.read_u64()?;
            let end_time = self.input.read_u64()?;
            assert_eq!(start_time, section.start_time);
            assert_eq!(end_time, section.end_time);

            // 66 is for the potential fastlz overhead
            let mem_required_for_traversal = self.input.read_u64()? + 66;
            let time_table = self.read_time_block(section.file_offset, section_length)?;

            todo!("DATA")
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_fst() {
        let f =
            std::fs::File::open("fsts/VerilatorBasicTests_Anon.fst").expect("failed to open file!");
        // read the header
        let mut headerReader = HeaderReader::new(f);
        headerReader.read().expect("It should work!");
        // read the actual data
        let (input, meta) = headerReader.into_input_and_meta_data().unwrap();
        DataReader::new(input, meta)
            .read()
            .expect("It should work!");
    }
}
