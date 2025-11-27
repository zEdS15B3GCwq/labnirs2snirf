# Documentation

This directory contains the Sphinx documentation for labnirs2snirf.

## Building the Documentation

### Using Nox (Recommended)

The easiest way to build the documentation is using Nox:

```bash
nox -s docs
```

This will:

1. Create a clean virtual environment
2. Install all required dependencies
3. Generate index.md by injecting the project readme into index.template
4. Build the HTML documentation
5. Output to `docs/_build/html/`

### Using Python module

```bash
python docs/generate_index.py
sphinx-build -b html docs/ docs/_build/
```

## Viewing the Documentation

After building, open `docs/_build/html/index.html` in your web browser.

## Documentation Structure

- `index.md` - Main documentation page with table of contents
- `installation.md` - Installation instructions
- `usage.md` - Usage guide and examples
- `api.md` - Auto-generated API reference from docstrings
- `contributing.md` - Contributing guidelines
- `conf.py` - Sphinx configuration

## Documentation Format

Markdown (`.md`) files with MyST parser for all documentation.
Python docstrings use NumPy format.

## Requirements

- sphinx
- sphinx-rtd-theme
- myst-parser

These are automatically installed when using `nox -s docs`.
