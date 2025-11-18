import pathlib
import pytest

@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def data_path():
    return pathlib.Path(__file__).parent / "data"
