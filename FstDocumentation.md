# FST Documentation

Available documentation:
- The [source code](https://github.com/gtkwave/gtkwave/tree/e1c01753bc5db9f7b42e41b9bde651a375ec5eba/gtkwave4/src/helpers/fst) of GTKWave.
- The [documentation](https://gtkwave.sourceforge.net/gtkwave.pdf) of GTKWave.
- An [unofficial specification](https://blog.timhutt.co.uk/fst_spec/) for FST format.


## FST API-Usage Study

### Verilator

Verilator uses the `fstWriter` API from [`include/verilated_fst_c.cpp`](https://github.com/verilator/verilator/blob/bd4eede6b47bc894f73ba6151f2ffe63db8feb3d/include/verilated_fst_c.cpp)

**Waveform**
- `fstWriterEmitValueChange`
- `fstWriterEmitTimeChange`
- `fstWriterFlushContext`

**Header / Meta-Data**
- `fstWriterSetTimescaleFromString`
- `fstWriterSetPackType(..., FST_WR_PT_LZ4)`
- `fstWriterSetParallelMode(..., 1)` (_optional_)

**Hierarchy**
- `fstWriterCreateVar`
- `fstWriterSetScope`
- `fstWriterSetUpscope`
- `fstWriterCreateEnumTable`
- `fstWriterEmitEnumTableRef`

**Open / Close**
- `fstWriterClose`
- `fstWriterCreate`

#### Variable Types / Direction

Information from [`verilator/src/V3EmitCImp.cpp`](https://github.com/verilator/verilator/blob/bd4eede6b47bc894f73ba6151f2ffe63db8feb3d/src/V3EmitCImp.cpp#L674)

**Types**
- `FST_VT_VCD_REAL_PARAMETER`
- `FST_VT_VCD_REAL`
- `FST_VT_VCD_PARAMETER`
- `FST_VT_VCD_SUPPLY0`
- `FST_VT_VCD_SUPPLY1`
- `FST_VT_VCD_TRI0`
- `FST_VT_VCD_TRI1`
- `FST_VT_VCD_TRI`
- `FST_VT_VCD_WIRE`
- `FST_VT_VCD_INTEGER`
- `FST_VT_SV_BIT`
- `FST_VT_SV_LOGIC`
- `FST_VT_SV_INT`
- `FST_VT_SV_SHORTINT`
- `FST_VT_SV_LONGINT`
- `FST_VT_SV_BYTE`
- `FST_VT_VCD_EVENT`

**Direction**
- `FST_VD_INOUT`
- `FST_VD_OUTPUT`
- `FST_VD_INPUT`
- `FST_VD_IMPLICIT`

### iVerilog

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
- `fstWriterSetPackType(..., 1)` (`FST_WR_PT_FASTLZ = 1`) (_optional_)
- `fstWriterSetRepackOnClose(..., 1)` (_optional_)

**Hierarchy**
- `fstWriterCreateVar`
- `fstWriterSetSourceInstantiationStem`
- `fstWriterSetSourceStem`
- `fstWriterSetScope`
- `fstWriterSetUpscope`

**Open / Close**
- `fstWriterClose`
- `fstWriterCreate`

### GHDL

GHDL uses the `fstWriter` API from [`src/grt/grt-fst.adb`](https://github.com/ghdl/ghdl/blob/b67ace3f4553e5072fb51d1de637e483cf56342a/src/grt/grt-fst.adb)

**Waveform**
- `fstWriterEmitValueChange`
- `fstWriterEmitVariableLengthValueChange`
- `fstWriterEmitTimeChange`

**Header / Meta-Data**
- `fstWriterSetVersion`
- `fstWriterSetTimescale`
- `fstWriterSetPackType(..., FST_WR_PT_LZ4)`
- `fstWriterSetRepackOnClose(..., 1)`
- `fstWriterSetFileType(..., FST_FT_VHDL)`
- `fstWriterSetParallelMode(..., 0)`

**Hierarchy**
- `fstWriterCreateVar2`
- `fstWriterSetSourceInstantiationStem`
- `fstWriterSetSourceStem`
- `fstWriterSetScope`
- `fstWriterSetUpscope`

**Open / Close**
- `fstWriterClose`
- `fstWriterCreate`

### Yosys

Yosys uses the `fstWriter` API from [`passes/sat/sim.cc`](https://github.com/YosysHQ/yosys/blob/417871e8319dbfbc27dabf0512c4dbd9fb9bf07d/passes/sat/sim.cc)

**Waveform**
- `fstWriterEmitValueChange`
- `fstWriterEmitTimeChange`

**Header / Meta-Data**
- `fstWriterSetDate`
- `fstWriterSetVersion`
- `fstWriterSetTimescaleFromString`
- `fstWriterSetPackType(..., FST_WR_PT_FASTLZ)`
- `fstWriterSetRepackOnClose(..., 1)`

**Hierarchy**
- `fstWriterCreateVar`
- `fstWriterSetScope`
- `fstWriterSetUpscope`

**Open / Close**
- `fstWriterClose`
- `fstWriterCreate`