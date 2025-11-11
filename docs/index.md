# labnirs2snirf Documentation

Welcome to the **labnirs2snirf** documentation!

This package provides tools to convert fNIRS experimental data exported from Shimadzu LabNIRS software to the Shared Near Infrared Spectroscopy Format (SNIRF), enabling compatibility with analysis tools like MNE-Python and other NIRS analysis software.

## Features

- Convert LabNIRS export files to SNIRF/HDF5 format
- Support for importing probe positions from layout files (.sfp format only, for now)
- Command-line interface and API
- Exclude data based on type and wavelength (e.g. only include raw data for 2 wavelengths)

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
usage
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
script
```

## Quick Start

Install the package:

```bash
pip install labnirs2snirf
```

Convert a LabNIRS file:

```bash
python -m labnirs2snirf input.txt output.snirf
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
