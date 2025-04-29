import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)
