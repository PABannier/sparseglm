from setuptools import setup, find_packages
from setuptools_rust import RustExtension

with open("./requirements.txt", "rt") as infile:
    install_requires = infile.read().splitlines()

setup(
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "sparseglm._lib",
            rustc_flags=["--cfg=Py_3"],
            args=["--no-default-features"],
        )
    ],
    long_description_content_type="text/markdown",
    url="https://github.com/PABannier/sparseglm",
    python_requires=">=3.6",
    install_requires=install_requires,
    packages=find_packages(),
)
