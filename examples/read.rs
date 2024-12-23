use fst_reader::FstReader;

fn main() {
    let input_path = std::env::args_os()
        .nth(1)
        .expect("missing input file argument");

    let input = std::fs::File::open(input_path).expect("failed to open input file!");

    let mut reader = FstReader::open(std::io::BufReader::new(input)).unwrap();

    let _header = reader.get_header();

    let mut ids = Vec::new();

    reader
        .read_hierarchy(|entry| match entry {
            fst_reader::FstHierarchyEntry::Var { handle, .. } => ids.push(handle),
            fst_reader::FstHierarchyEntry::Scope { .. } => {}
            fst_reader::FstHierarchyEntry::UpScope { .. } => {}
            _ => {}
        })
        .unwrap();

    let filter = fst_reader::FstFilter::new(0, u64::MAX, ids);

    reader
        .read_signals(&filter, |time_idx, handle, value| {
            std::hint::black_box((time_idx, handle, value));
        })
        .unwrap();
}
