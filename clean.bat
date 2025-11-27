@echo off
del .coverage 2>nul
del labnirs2snirf.log 2>nul
del pysnirf2.log 2>nul
del coverage.xml 2>nul
del docs/index.md 2>nul
rmdir /s /q .mypy_cache 2>nul
rmdir /s /q .nox 2>nul
rmdir /s /q .pytest_cache 2>nul
rmdir /s /q .ruff_cache 2>nul
rmdir /s /q .venv 2>nul
rmdir /s /q __pycache__ 2>nul
rmdir /s /q dist 2>nul
rmdir /s /q htmlcov 2>nul
rmdir /s /q docs\build 2>nul
rmdir /s /q docs\__pycache__ 2>nul
rmdir /s /q labnirs2snirf\__pycache__ 2>nul
rmdir /s /q tests\__pycache__ 2>nul
