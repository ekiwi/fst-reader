use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::io::{Read, Seek, SeekFrom};
use std::str::Utf8Error;
use std::sync::atomic::spin_loop_hint;

pub fn add(left: usize, right: usize) -> usize {
    left + right
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

const DOUBLE_ENDIAN_TEST: f64 = 2.7182818284590452354;

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
    file_type: u8,
    time_zero: u64,
}

#[derive(Debug)]
enum ReaderErrorKind {
    IO(std::io::Error),
    FromPrimitive(),
    StringParse(Utf8Error),
    ParseVariant(),
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

impl From<Utf8Error> for ReaderError {
    fn from(value: Utf8Error) -> Self {
        let kind = ReaderErrorKind::StringParse(value);
        ReaderError { kind }
    }
}

type Result<T> = std::result::Result<T, ReaderError>;

struct Reader<R: Read + Seek> {
    input: R,
    header: Option<Header>,
    buf: [u8; 128], // used for reading
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
        res = (res << 7) | (bb & 0x7f)
    }
    Ok((res, len))
}

impl<R: Read + Seek> Reader<R> {
    fn new(input: R) -> Self {
        Reader {
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

    fn read_string(&mut self, len: usize) -> Result<String> {
        self.input.read_exact(&mut self.buf[..len])?;
        let zero_index = self.buf.iter().position(|b| *b == 0u8).unwrap_or(len - 1);
        Ok((std::str::from_utf8(&self.buf[..(zero_index + 1)])?).to_string())
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
        let file_type = self.read_u8()?;
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

    fn read_bv_data_dynamic_alias_2(&mut self) -> Result<()> {
        // this section only seems to matter if the header is incomplete
        let section_length = self.read_u64()?;
        let bt = self.read_u64()?;
        let end_time = self.read_u64()?;
        // TODO: skip rest
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

        assert_eq!(
            compressed_length, uncompressed_length,
            "TODO: add decompression!"
        );

        todo!();

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
                BlockType::VcData => self.read_bv_data_dynamic_alias_2()?,
                BlockType::VcDataDynamicAlias => self.read_bv_data_dynamic_alias_2()?,
                BlockType::VcDataDynamicAlias2 => self.read_bv_data_dynamic_alias_2()?,
                BlockType::Blackout => {}
                BlockType::Geometry => self.read_geometry()?,
                BlockType::Hierarchy => {}
                BlockType::HierarchyLZ4 => {}
                BlockType::HierarchyLZ4Duo => {}
                BlockType::GZipWrapper => {}
                BlockType::Skip => {}
            };
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
        Reader::new(f).read().expect("It should work!");
    }
}
