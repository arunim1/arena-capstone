from pathlib import Path

from setuptools import find_packages, setup

if Path("requirements.txt").exists():
    requirements = Path("requirements.txt").read_text("utf-8").splitlines()
else:
    requirements = []

setup(
    name="arena-capstone",
    version="0.0.1",
    description="",
    long_description=Path("README.md").read_text("utf-8"),
    author="Arumin, Connor, Glen, Jay, Spencer",
    author_email="",
    url="https://github.com/arunim1/arena-capstone",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
