# setup.py at project root
from setuptools import setup, find_packages

setup(
    name="UniEval",
    version="0.1",
    packages=find_packages(where="src/evaluation"),
    package_dir={"": "src/evaluation"},
    install_requires=[
    ],
)
