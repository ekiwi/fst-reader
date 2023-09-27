use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::fs::read;
use std::io::{Read, Seek, SeekFrom};

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

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
    input: R,
    header: Option<Header>,
    buf: [u8; 128], // used for reading
}

struct HierarchyReader<R: Read> {
    input: R,
}

#[inline]
fn get_variant_32(bytes: &[u8]) -> Result<(u32, usize)> {
    // find end byte (with bit 7 cleared)
    let len = match bytes.iter().position(|b| (b & 0x80) == 0) {
        None => {
            return Err(ReaderError {
                kind: ReaderErrorKind::ParseVariant(),
            })
        }
        Some(end_index) => end_index + 1,
    };
    // read bits, 7 at a time
    let mut res = 0u32;
    for bb in bytes.iter().take(len).rev() {
        res = (res << 7) | ((*bb as u32) & 0x7f)
    }
    Ok((res, len))
}

#[inline]
fn read_variant_32(input: &mut impl Read) -> Result<u32> {
    let mut res = 0u32;
    for ii in 0..8 {
        let byte = read_u8(input)?;
        let value = (byte as u32) & 0x7f;
        res |= value << (7 * ii);
        if (byte & 0x80) == 0 {
            return Ok(res);
        }
    }
    Err(ReaderError {
        kind: ReaderErrorKind::ParseVariant(),
    })
}

#[inline]
fn read_u8(input: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_c_str(input: &mut impl Read, max_len: usize) -> Result<String> {
    let mut bytes: Vec<u8> = Vec::with_capacity(32);
    loop {
        let byte = read_u8(input)?;
        if byte == 0 {
            break;
        } else {
            bytes.push(byte);
        }
    }
    Ok(String::from_utf8(bytes)?)
}

const HIERARCHY_NAME_MAX_SIZE: usize = 512;
const HIERARCHY_ATTRIBUTE_MAX_SIZE: usize = 65536 + 4096;

impl<R: Read> HierarchyReader<R> {
    fn new(input: R) -> Self {
        HierarchyReader { input }
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
        let entry_tpe = match read_u8(&mut self.input) {
            Ok(tpe) => tpe,
            Err(e) => return Ok(None),
        };
        let entry = match entry_tpe {
            254 => {
                // VcdScope (ScopeType)
                let tpe = ScopeType::try_from_primitive(read_u8(&mut self.input)?)?;
                let name = read_c_str(&mut self.input, HIERARCHY_NAME_MAX_SIZE)?;
                let component = read_c_str(&mut self.input, HIERARCHY_NAME_MAX_SIZE)?;
                HierarchyEntry::Scope {
                    tpe,
                    name,
                    component,
                }
            }
            0..=29 => {
                // VcdEvent ... SvShortReal (VariableType)
                let tpe = VarType::try_from_primitive(entry_tpe)?;
                let direction = VarDirection::try_from_primitive(read_u8(&mut self.input)?)?;
                let name = read_c_str(&mut self.input, HIERARCHY_NAME_MAX_SIZE)?;
                let raw_length = read_variant_32(&mut self.input)?;
                let length = if tpe == VarType::Port {
                    // remove delimiting spaces and adjust signal size
                    (raw_length - 2) / 3
                } else {
                    raw_length
                };
                let alias = read_variant_32(&mut self.input)?;
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
            input,
            header: None,
            buf: [0u8; 128],
        }
    }

    fn header_incomplete(&self) -> bool {
        match &self.header {
            None => true,
            Some(h) => h.start_time == 0 && h.end_time == 0,
        }
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

    fn read_header(&mut self) -> Result<()> {
        let section_length = self.read_u64()?;
        assert_eq!(section_length, 329);
        let start_time = self.read_u64()?;
        let end_time = self.read_u64()?;
        let endian_test = self.read_f64()?;
        assert_eq!(endian_test, DOUBLE_ENDIAN_TEST);
        let memory_used_by_writer = self.read_u64()?;
        let scope_count = self.read_u64()?;
        let var_count = self.read_u64()?;
        let max_var_id_code = self.read_u64()?;
        let vc_section_count = self.read_u64()?;
        let timescale_exponent = self.read_i8()?;
        let version = self.read_string(128)?;
        // this size was reduced compared to what is documented in block_format.txt
        let date = self.read_string(119)?;
        let file_type = FileType::try_from(self.read_u8()?)?;
        let time_zero = self.read_u64()?;

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

    fn read_data(&mut self) -> Result<()> {
        // this is the data section
        let section_length = self.read_u64()?;
        let bt = self.read_u64()?;
        let end_time = self.read_u64()?;
        if self.header_incomplete() {
            todo!("Fixup missing header with info from data section!")
        }
        self.skip(section_length, 3 * 8)?;
        Ok(())
    }

    fn skip(&mut self, section_length: u64, already_read: i64) -> Result<u64> {
        Ok(self
            .input
            .seek(SeekFrom::Current((section_length as i64) - already_read))?)
    }

    fn read_geometry(&mut self) -> Result<()> {
        let section_length = self.read_u64()?;
        // skip this section if the header is not complete
        if self.header_incomplete() {
            self.skip(section_length, 8)?;
            return Ok(());
        }

        let uncompressed_length = self.read_u64()?;
        let max_handle = self.read_u64()?;
        let compressed_length = section_length - 3 * 8;

        let bytes = if compressed_length == uncompressed_length {
            self.read_bytes(uncompressed_length as usize)?
        } else {
            todo!("add support for decompression")
        };

        println!("max_handle = {max_handle}");
        let mut byte_ii = 0usize;
        let mut longest_signal_value_len = 32;
        for ii in 0..max_handle {
            let (value, inc) = get_variant_32(&bytes[byte_ii..])?;
            byte_ii += inc;
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
            println!("{ii} {length} {tpe:?}")
        }

        Ok(())
    }

    fn read_hierarchy_bytes(&mut self, compression: HierarchyCompression) -> Result<Vec<u8>> {
        // Note: the GTKWave implementation skips decoding the hierarchy block initially
        // it also first re-creates a history file and then read from it.
        // Here we only recreate things in memory.
        // This function is similar to fstReaderRecreateHierFile

        let section_length = self.read_u64()? as usize;
        let uncompressed_length = self.read_u64()? as usize;
        let compressed_length = section_length - 2 * 8;

        let bytes = match compression {
            HierarchyCompression::ZLib => todo!("ZLib compression is currently not supported!"),
            HierarchyCompression::Lz4 => {
                let compressed = self.read_bytes(compressed_length)?;
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
            let block_tpe = match self.read_block_tpe() {
                Err(_) => break,
                Ok(tpe) => tpe,
            };
            println!("{block_tpe:?}");
            match block_tpe {
                BlockType::Header => self.read_header()?,
                BlockType::VcData => self.read_data()?,
                BlockType::VcDataDynamicAlias => self.read_data()?,
                BlockType::VcDataDynamicAlias2 => self.read_data()?,
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
}

// TODO: get actual data
// look at fstReaderIterBlocks2 and fstReaderGetFacProcessMask

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_fst() {
        let f =
            std::fs::File::open("fsts/VerilatorBasicTests_Anon.fst").expect("failed to open file!");
        HeaderReader::new(f).read().expect("It should work!");
    }
}
