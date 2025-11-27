# labnirs2snirf

[![ci](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/ci.yml/badge.svg)](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/ci.yml)
[![CodeQL](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/github-code-scanning/codeql)
[![Coverage](https://img.shields.io/endpoint?url=https://zEdS15B3GCwq.github.io/labnirs2snirf/badges/coverage.json)](https://zEdS15B3GCwq.github.io/labnirs2snirf/htmlcov/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://zEdS15B3GCwq.github.io/labnirs2snirf/)
[![python support](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-blue?logo=python)](https://test.pypi.org/project/labnirs2snirf/)

<!-- INDEX_START -->

This package provides tools to convert fNIRS experimental data exported from
Shimadzu LabNIRS software to the Shared Near Infrared Spectroscopy Format
(SNIRF), enabling compatibility with analysis tools like MNE-Python and other
NIRS analysis software.

## Features

- Convert LabNIRS export files to SNIRF/HDF5 format
- Your data is untouched (no unnecessary Hb -> OD -> raw conversion)
- Support for importing probe positions from layout file (.sfp format only, for
  now)
- Command-line interface and API
- Exclude data based on type and wavelength (e.g. only include raw data for 2
  wavelengths), to allow 3rd-party tools with specific requirements to read the
  file.

## Motivation

Most popular analytical tools (Homer, MNE, etc.) do not support reading LabNirs
data files and LabNirs has no official converter either. The recommended way for
conversion, as far as I know, is to use the `Shimadzu2nirs.m` Matlab script to
convert to .nirs format that Homer can import and save as .snirf.

However, the script takes haemoglobin concentration data and converts it back to
OD and then to intensity, and I would prefer to use untouched raw data directly
without any unnecessary transformation steps. I also just simply prefer to avoid
using Matlab where possible.

This tool aims to simplify the process. Data exported from LabNirs (Hb or
voltage or both) can be converted to .snirf in one step, without unnecessary
conversion.

## Current state of the project

The conversion tool is functional, with the following caveats:

1. The expected file format is based on data from one single LabNIRS machine.
   Other machines or versions of the LabNIRS software could have a different
   output format, which is likely to lead to errors.

2. I only use the recording mode in which task changes are marked by event
   triggers. Metadata fields may have a different meaning in other modi of
   operandi.

3. Analysis tools (e.g. MNE, Homer) sometimes have their own limitations
   regarding snirf data format, and while **labnirs2snirf** produces valid
   .snirf files, those tools may not be able to read just any valid file. To
   help with this, it is possible to limit what is included in the .snirf file
   (e.g. only 2 wavelengths, only Hb or voltage data) using the `--type` and
   `--drop` options.

4. LabNirs stores more metadata (e.g. task duration, pre and post rest periods)
   in additional files, that is not exported. In theory, I may implement reading
   metadata from those files in the future, but no guarantees. It may be easier
   to just add the missing information using other software.

5. For now, probe location data can only be supplied in the .sfp file format. I
   may add support for more formats later. It is also possible not to include
   layout data (in which case all probes have all-zero coordinates), but that
   will not make 3rd party tools happy.

## Quick Start

```bash
# 1. install labnirs2snirf from PyPI
pip install labnirs2snirf

# 2. step into the project folder
cd labnirs2snirf

# 3. create a Python virtual environment
python -m venv .venv

# 4. activate the virtual environment
./.venv/Scripts/activate

# 5. install labnirs2snirf and dependencies
python -m pip install .

# 6. run without parameters or with --help to print out usage instructions
labnirs2snirf
labnirs2snirf --help

# 7. convert labnirs file to .snirf
# You can use one of the data files on the project repository for testing, e.g.
# https://raw.githubusercontent.com/zEdS15B3GCwq/labnirs2snirf/refs/heads/main/data/test/small_labnirs.txt
labnirs2snirf small_labnirs.txt small.snirf
```

<!-- INDEX_END -->

## Documentation

Check the [documentation](https://zEdS15B3GCwq.github.io/labnirs2snirf/) for
further details.
