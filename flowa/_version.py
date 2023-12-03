"""
flowa._version

Module containing the version info of Flowa.
"""

PYTHON_REQUIRED_MAJOR = 3
PYTHON_REQUIRED_MINOR = 7


def get_python(*args, **kwargs) -> tuple:
    return (PYTHON_REQUIRED_MAJOR, PYTHON_REQUIRED_MINOR)


def get_version(*args, **kwargs) -> dict:
    """Returns the version of Flowa."""
    version: dict = {
        "PYTHON_MAJOR": 3,
        "PYTHON_MINOR": 7,
        "PYTHON_PATCH": 1,
        "PYTHON_VERSION": 3.7,
        "FLOWA_MAJOR": 1,
        "FLOWA_MINOR": 1,
        "FLOWA_PATCH": 6,
        "FLOWA_VERSION": "1.1.7",
    }
    return version
