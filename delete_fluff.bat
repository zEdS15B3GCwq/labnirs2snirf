@echo off
del .coverage
del labnirs2snirf.log
del pysnirf2.log
del coverage.xml
rmdir /s /q .mypy_cache
rmdir /s /q .nox
rmdir /s /q .pytest_cache
rmdir /s /q .ruff_cache
rmdir /s /q .venv
rmdir /s /q __pycache__
rmdir /s /q dist
rmdir /s /q htmlcov
rmdir /s /q docs\build
rmdir /s /q labnirs2snirf\__pycache__
rmdir /s /q tests\__pycache__
