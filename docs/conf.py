"""Configure Sphinx."""

import os
import sys

# Add the project root to the path so Sphinx can find your package
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "labnirs2snirf"
copyright = "2025, Tamas Fehervari"  # pylint: disable=redefined-builtin
author = "Tamas Fehervari"
release = "0.1.0"  # Use your actual version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Napoleon settings for numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# HTML theme
html_theme = "sphinx_rtd_theme"  # or 'alabaster', 'classic', etc.

# Add any paths that contain templates here
templates_path = []  # type: ignore # ["_templates"]

# List of patterns to exclude
exclude_patterns = ["__pycache__", "_build"]

# Configure MyST to parse markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
