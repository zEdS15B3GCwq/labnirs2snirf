"""Configure Sphinx."""

import os
import re
import sys
from pathlib import Path

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


def _inject_readme_into_index() -> None:
    """Generate index.md by injecting readme.md into index.template."""
    project_root = Path(__file__).parents[1]
    readme_path = project_root / "README.md"
    template_path = Path(__file__).parent / "index.template"
    if not (readme_path.exists() and template_path.exists()):
        return

    readme_text = readme_path.read_text(encoding="utf8")
    body = re.search(
        r"(?<=<!-- INDEX_START -->\n).*(?=\n<!-- INDEX_END -->)",
        readme_text,
        flags=re.S,
    ).group(0)

    template_text = template_path.read_text(encoding="utf8")
    index_text = re.sub(
        r"<!-- README_START -->.*?<!-- README_END -->",
        body,
        template_text,
        flags=re.S,
    )

    index_path = Path(__file__).parent / "index.md"
    index_path.write_text(index_text, encoding="utf8")


# perform injection at import/build time
if __name__ == "__main__":
    _inject_readme_into_index()
