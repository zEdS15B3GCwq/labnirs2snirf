# labnirs2snirf

[![ci](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://zEdS15B3GCwq.github.io/labnirs2snirf/badges/coverage.json)](https://zEdS15B3GCwq.github.io/labnirs2snirf/htmlcov/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://zEdS15B3GCwq.github.io/labnirs2snirf/)

This package provides tools to convert fNIRS experimental data exported from
Shimadzu LabNIRS software to the Shared Near Infrared Spectroscopy Format (SNIRF),
enabling compatibility with analysis tools like MNE-Python and other NIRS analysis
software.

## Features

- Convert LabNIRS export files to SNIRF/HDF5 format
- Support for importing probe positions from layout files
(.sfp format only, for now)
- Command-line interface and API
- Exclude data based on type and wavelength (e.g. only include raw data for
2 wavelengths), to allow 3rd-party tools with specific requirements to read
the file.

## Quick Start

Once the package is published on PyPI, the installation will be as simple as
`pip install labnirs2snirf`.

For now, you need to download the source files either by cloning this repository
or downloading a .zipped package.

```bash
# 1. clone the files from the repository (or download a .zip)
git clone https://github.com/zEdS15B3GCwq/labnirs2snirf.git

# 2. step into the project folder
cd labnirs2snirf

# 3. create a Python virtual environment
python -m venv .venv

# 4. activate the virtual environment
./.venv/Scripts/activate

# 5. install labnirs2snirf and dependencies
python -m pip install .

# 6. run without parameters to print out usage instructions
python -m labnirs2snirf

# 7. convert labnirs file to .snirf
python -m labnirs2snirf tests/small_labnirs.txt small.snirf
```

## Documentation

Check the [documentation](https://zEdS15B3GCwq.github.io/labnirs2snirf/)
for further details.
