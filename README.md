# labnirs2snirf

[![ci](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/zEdS15B3GCwq/labnirs2snirf/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://zEdS15B3GCwq.github.io/labnirs2snirf/badges/coverage.json)](https://zEdS15B3GCwq.github.io/labnirs2snirf/htmlcov/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://zEdS15B3GCwq.github.io/labnirs2snirf/)

This package provides tools to convert fNIRS experimental data exported from Shimadzu LabNIRS software to the Shared Near Infrared Spectroscopy Format (SNIRF), enabling compatibility with analysis tools like MNE-Python and other NIRS analysis software.

## Features

- Convert LabNIRS export files to SNIRF/HDF5 format
- Support for importing probe positions from layout files (.sfp format only, for now)
- Command-line interface and API
- Exclude data based on type and wavelength (e.g. only include raw data for 2 wavelengths), to allow 3rd-party tools with specific requirements to read the file.

## Quick Start

Once the package is published on PyPI, the installation will be as simple as `pip install labnirs2snirf`.

For now, you need to install from a cloned repository.

1. Clone the repository:

```bash
git clone https://zEdS15B3GCwq.github.io/labnirs2snirf/
```

Or download and extract the zipped repository from GitHub.

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
./.venv/Scripts/activate
```

3. Install labnirs2snirf into the new virtual environment:

```bash
python -m pip install .
```

4. Convert a LabNIRS file:

```bash
python -m labnirs2snirf input.txt output.snirf
```

## Documentation

Check the [documentation](https://zEdS15B3GCwq.github.io/labnirs2snirf/) for further details.
