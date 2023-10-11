# FST Documentation

Available documentation:
- The [source code](https://github.com/gtkwave/gtkwave/tree/e1c01753bc5db9f7b42e41b9bde651a375ec5eba/gtkwave4/src/helpers/fst) of GTKWave.
- The [documentation](https://gtkwave.sourceforge.net/gtkwave.pdf) of GTKWave.
- An [unofficial specification](https://blog.timhutt.co.uk/fst_spec/) for FST format.


## FST API-Usage Study

### iVerilog Usage

iVerilog uses the `fstWriter` API from [`vpi/sys_fst.c`](https://github.com/steveicarus/iverilog/blob/c498d53d0d6565ec607e5cc472c1d58f58810d52/vpi/sys_fst.c)

**Waveform**
- `fstWriterEmitValueChange`
- `fstWriterEmitTimeChange`
- `fstWriterEmitDumpActive` for `$dumpon` and `$dumpoff`
- `fstWriterFlushContext`
- `fstWriterGetDumpSizeLimitReached`
- `fstWriterSetDumpSizeLimit`

**Header / Meta-Data**
- `fstWriterSetDate`
- `fstWriterSetVersion`
- `fstWriterSetTimescaleFromString`
- `fstWriterSetPackType(..., 1)` (`FST_WR_PT_FASTLZ = 1`)
- `fstWriterSetRepackOnClose(..., 1)` creates a GZip wrapper
- 

**Hierarchy**
- `fstWriterCreateVar`
- `fstWriterSetSourceInstantiationStem`
- `fstWriterSetSourceStem`
- `fstWriterSetScope`
- `fstWriterSetUpscope`

**Open / Close** 

- `fstWriterClose`
- `fstWriterCreate`
