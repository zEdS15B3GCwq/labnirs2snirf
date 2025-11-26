# Installation

## Requirements

- Python 3.12 or higher
- pip package manager

## Install from Source

To install the latest development version from source:

```bash
git clone git clone https://github.com/zEdS15B3GCwq/labnirs2snirf.git
cd labnirs2snirf
pip install .
```

## Install from PyPI (not published yet)

Install labnirs2snirf using pip:

```bash
pip install labnirs2snirf
```

## Verify Installation

To verify the installation was successful you can run any of these:

```bash
# invoke using the script shortcut, should print usage instructions
labnirs2snirf -h

# run package as a script, should print usage instructions
python -m labnirs2snirf

# test if module can be imported for programmatic use
python -c "import labnirs2snirf; print(labnirs2snirf.__name__)"
```

## Dependencies

The package has the following core dependencies:

- **polars** (>=1.24.0) - DataFrame operations for data processing
- **h5py** (>=3.13.0) - HDF5 file format support for SNIRF output
