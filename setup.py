# setup.py at project root
from setuptools import setup, find_packages

setup(
    name="UniEval",
    version="0.1",
    # 在 src/evaluation 目录下查找所有包
    packages=find_packages(where="src/evaluation"),
    # 映射包名称到物理目录
    package_dir={"": "src/evaluation"},
    install_requires=[
        # 如果有依赖，这里可以列
    ],
)
