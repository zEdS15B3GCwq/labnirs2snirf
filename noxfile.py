"""Nox sessions for linting, testing, and documentation."""

import nox

# Python versions to test against
PYTHON_ALL_VERSIONS = ["3.12", "3.13", "3.14"]
PYTHON_MAIN_VERSION = "3.14"
PYTHON_OTHER_VERSIONS = list(set(PYTHON_ALL_VERSIONS) - {PYTHON_MAIN_VERSION})

# Alternative way to get Python versions from pyproject.toml
# Load project metadata from pyproject.toml
# PYPROJECT = nox.project.load_toml("pyproject.toml")
# DEPENDENCIES = PYPROJECT["project"]["dependencies"]
# PYTHON_VERSIONS = nox.project.python_versions(PYPROJECT, max_version="3.14")

# Default sessions to run when no session is explicitly specified
nox.options.sessions = [
    "tests_with_coverage",
    "lint",
    "type_check",
    "fmt_check",
    "install_test",
]

# Use UV virtual environment backend by default
nox.options.default_venv_backend = "uv|virtualenv"

# Disable automatic Python downloads by nox, which is enabled by the UV backend
# nox.options.download_python = "never"


@nox.session(python=PYTHON_OTHER_VERSIONS)
def tests(session):
    """Run the test suite with pytest."""
    session.install("-e", ".")
    session.install("pytest", "snirf")

    # Run pytest with any additional arguments passed via command line
    session.run("pytest", "tests", *session.posargs)


@nox.session(python=PYTHON_MAIN_VERSION, tags=["pre-commit"])
def tests_main_python(session):
    """Run the test suite with pytest on the main Python version."""
    session.install("-e", ".")
    session.install("pytest", "snirf")
    session.run(
        "pytest",
        "tests",
    )


@nox.session(python=PYTHON_MAIN_VERSION)
def tests_with_coverage(session):
    """Run tests with coverage reporting."""
    session.install("-e", ".")
    session.install("pytest", "pytest-cov", "snirf")

    # Run pytest with coverage
    session.run(
        "pytest",
        "--cov=labnirs2snirf",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "tests/",
        *session.posargs,
    )


@nox.session(python=PYTHON_MAIN_VERSION)
def lint(session):
    """Run linting checks with ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")


@nox.session(python=PYTHON_MAIN_VERSION)
def type_check(session):
    """Run type checking with mypy."""
    session.install("-e", ".")
    session.install("mypy", "pytest", "h5py")
    session.run("mypy", "labnirs2snirf", "tests")


@nox.session(python=PYTHON_MAIN_VERSION)
def fmt_check(session):
    """Check code formatting with ruff."""
    session.install("ruff")
    session.run("ruff", "format", "--check", "--diff", ".")


@nox.session(python=PYTHON_MAIN_VERSION)
def fmt(session):
    """Auto-format code with ruff."""
    session.install("ruff")
    session.run("ruff", "format", ".")


@nox.session(python=PYTHON_MAIN_VERSION)
def docs(session):
    """Build the documentation with Sphinx."""
    session.install("-e", ".")
    session.install("sphinx", "sphinx-rtd-theme", "myst-parser")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")


@nox.session(python=PYTHON_ALL_VERSIONS, venv_backend="venv")
def install_test(session):
    """Test that the package can be installed cleanly in a fresh environment."""
    session.install(".")
    session.run("python", "-c", "import labnirs2snirf; print(labnirs2snirf.__name__)")


@nox.session(python=PYTHON_MAIN_VERSION)
def docs_clean(session):
    """Clean the documentation build directory."""
    import shutil
    from pathlib import Path

    build_dir = Path("docs/_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        session.log("Cleaned documentation build directory")
    else:
        session.log("Documentation build directory does not exist")
