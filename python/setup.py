from setuptools import find_packages, setup
from setuptools_rust import RustExtension


with open("./requirements.txt", "rt") as infile:
    install_requires = infile.read().splitlines()

setup(
    name="rustylasso",
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "rustylasso._lib",
            rustc_flags=["--cfg=Py_3"],
            features=["numpy/python3"],
            args=["--no-default-features"],
        )
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    packages=find_packages(),
)
