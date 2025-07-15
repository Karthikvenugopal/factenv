from setuptools import setup, find_packages

setup(
    name="factenv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=1.1.1",
        "torch>=2.7.0",
        "transformers>=4.52.0",
        "langchain>=0.3.25",
        "langchain-openai>=0.3.18",
        "langgraph>=0.4.8",
        "numpy>=1.26.0",
    ],
    python_requires=">=3.10",
)
