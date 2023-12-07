"""
flowa._version

Module containing the version info of Flowa.
"""

PYTHON_REQUIRED_MAJOR = 3
PYTHON_REQUIRED_MINOR = 7

FLOWA_MAJOR = 1
FLOWA_MINOR = 2
FLOWA_PATCH = 3

def get_python(*args, **kwargs) -> tuple:
    return (PYTHON_REQUIRED_MAJOR, PYTHON_REQUIRED_MINOR)


def get_version(*args, **kwargs) -> dict:
    """Returns the version of Flowa."""
    version: dict = {
        "PYTHON_MAJOR": PYTHON_REQUIRED_MAJOR,
        "PYTHON_MINOR": PYTHON_REQUIRED_MINOR,
        "PYTHON_PATCH": 1,
        "PYTHON_VERSION": f"{PYTHON_REQUIRED_MAJOR}.{PYTHON_REQUIRED_MINOR}.1",
        "FLOWA_MAJOR": FLOWA_MAJOR,
        "FLOWA_MINOR": FLOWA_MINOR,
        "FLOWA_PATCH": FLOWA_PATCH,
        "FLOWA_VERSION": f"{FLOWA_MAJOR}.{FLOWA_MINOR}.{FLOWA_PATCH}",
    }
    return version
