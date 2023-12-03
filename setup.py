"""
Setup script for flowa

Variables:
    path_absolute (str): Absolute path to the directory containing the setup.py file
    version (str): Current version of the package

Functions:
    setup: Setup function for the package
"""

from setuptools import setup, find_packages
from pathlib import Path

path_absolute: Path = Path(__file__).parent.absolute()

with open(f"{path_absolute}/flowa/__init__.py", "r") as file:
    for line in file.readlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip()[1:-1]
            break
 
setup(
    name="flowa",
    version=version,
    description="flowa - Decision Trees & Label Encoding",
    long_description=Path(f"{path_absolute}/README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/flowa-ai/flowa",
    author='flowa (Discord: @flow.a)',
    author_email='flowa.dev@gmail.com',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Typing :: Typed",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">=3.7",
    keywords=["flowa"],
)
