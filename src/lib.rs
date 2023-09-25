use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use std::io::Error;

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

struct Header {}

#[derive(Debug)]
enum ReaderErrorKind {
    IO(std::io::Error),
    FromPrimitive(),
}
#[derive(Debug)]
struct ReaderError {
    kind: ReaderErrorKind,
}

impl From<std::io::Error> for ReaderError {
    fn from(value: Error) -> Self {
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

type Result<T> = std::result::Result<T, ReaderError>;

struct Reader<R: std::io::Read> {
    input: R,
    buf: [u8; 8], // used for reading
}

impl<T: std::io::Read> Reader<T> {
    fn new(input: T) -> Self {
        Reader {
            input,
            buf: [0u8; 8],
        }
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.input.read_exact(&mut self.buf)?;
        Ok(u64::from_be_bytes(self.buf))
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.input.read_exact(&mut self.buf[..1])?;
        Ok(self.buf[0])
    }

    fn read_block_tpe(&mut self) -> Result<BlockType> {
        Ok(BlockType::try_from(self.read_u8()?)?)
    }

    fn read_header(&mut self) -> Result<()> {
        Ok(())
    }

    fn read(&mut self) -> Result<()> {
        let block_tpe = self.read_block_tpe()?;
        println!("{block_tpe:?}");
        match block_tpe {
            BlockType::Header => self.read_header()?,
            BlockType::VcData => {}
            BlockType::Blackout => {}
            BlockType::Geometry => {}
            BlockType::Hierarchy => {}
            BlockType::VcDataDynamicAlias => {}
            BlockType::HierarchyLZ4 => {}
            BlockType::HierarchyLZ4Duo => {}
            BlockType::VcDataDynamicAlias2 => {}
            BlockType::GZipWrapper => {}
            BlockType::Skip => {}
        };

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
