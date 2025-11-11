# Usage Guide

## Examples

The simplest way to use labnirs2snirf is through its command-line interface.

### Basic Conversion

Convert a LabNIRS export file to SNIRF format:

```bash
python -m labnirs2snirf input.txt output.snirf
```

### With Probe Layout

If you have a probe layout file (.sfp format), you can include it in the conversion:

```bash
python -m labnirs2snirf input.txt --layout layout.sfp
```

This writes the data to the default target file `out.snirf`.
Support for other probe layout formats may be implemented in the future.

### Selecting raw voltage data only

Suppose that you have a LABNIRS file with both raw voltage and Hb data, and want to export only the former:

```bash
python -m labnirs2snirf input.txt --type raw
```

The `--type` option specifies which data type (raw, Hb, all) to keep in the output.

### Keeping HbO and HbR only

Suppose that you have a LABNIRS file with both raw voltage and Hb data, and want to export only HbO and HbR data:

```bash
python -m labnirs2snirf input.txt --type hb --drop hbt
```

The `--type hb` option keeps only haemoglobin data, while `--drop hbt` excludes total Hb. If the LABNIRS file only contains Hb data, `--type` is unnecessary.

## Command-Line Options

List parameters and available options:

```bash
python -m labnirs2snirf -h
python -m labnirs2snirf --help
```

Available options:

- `input`: Path to the input LabNIRS text file (required)
- `output`: Path to the output SNIRF file (optional, default: "out.snirf")
- `--locations`: Path to probe layout/montage file (optional)
- `--type`: Select type of data to include (possible values: hb, raw, all)
- `--drop`: Exclude specific data type (HbO, HbR, HbT or wavelength); can be used multiple times
- `--verbose`,`-v`: Enable verbose logging output; can be repeated up to 3 times (e.g. -vvv)
- `--log`: Redirects log output to "labnirs2snirf.log"; implies at least one level of verbosity

## Python API

Use labnirs2snirf programmatically in Python code.

### Example

```python
from pathlib import Path
from labnirs2snirf.labnirs import read_labnirs
from labnirs2snirf.layout import read_layout, update_layout
from labnirs2snirf.snirf import write_snirf

# Read in data, keep only HbO and HbR
data = read_labnirs(
    data_file=Path("input.txt"),
    keep_category="hb",
    drop_subtype=["HbT"],
)

# Add probe layout data from file
layout = read_layout(Path("layout.sfp"))
update_layout(data, layout)

# Export to output.snirf
write_snirf(data, Path("output.snirf"))
```

## File Formats

### Input File Format

Text file exported by the LabNIRS software. LABNIRS allows to select whether to include _voltage_ and/or _raw_
data, and obviously, if one data type is not included, the relevant `--drop` and `--type` switches may not
work as intended. It is possible to run into an error for not having any data when dropping incorrect data types.

### Output Format

The output is a valid SNIRF (HDF5) file. Validity can be verified with the [snirf](https://github.com/BUNPC/pysnirf2) library.

### Probe Layout Files

Probe layout files specify the 3D positions of sources and detectors. The expected format is
a .sfp file, which is a tab-separated text file with columns:

- Source/Detector Label (e.g. S1)
- X, Y, Z coordinates

Example layout.sfp:
```
S1\t10.0\t0.0\t5.0
D1\t15.0\t0.0\t1.0
```

## Limitations

### Missing Metadata

LABNIRS does not export all data relevant to the experiment:

- Additional "patient" and "study" metadata (kept in a `.pat` file)
- Task duration, pre and post rest times (listed in a `.csv` file)
- Probe layout

Currently, **labnirs2snirf** is only able to read the exported data, not these additional files.
This means, that only the task onset is written correctly to SNIRF, duration is set to 0,
and other timings are not stored.

### Input File Version / Format

So far I have only seen export files with header version 11 and 35 lines of header data.
If other formats exist, **labnirs2snirf** may not work with them.

### Tasks

I am mostly familiar with using the EVENT marker for indicating TASK changes, and
have not really looked into how other operating modes work. There may be problems with
how task-related data is interpreted by **labnirs2snirf** in such cases.

If you think you have encountered a problem possibly due to these limitations,
you're welcome to open a GitHub issue and describe in detail what happened.
