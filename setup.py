from setuptools import setup, find_packages

setup(
    name="factenv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=1.1.1",
        "numpy>=1.26.0",
    ],
    python_requires=">=3.10",
)
