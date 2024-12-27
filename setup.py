from setuptools import setup, find_packages

setup(
    name="synth_pack",
    version="0.1.1",
    description="Modular Synthesizer Package for Python",
    author="Shin",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "soundfile",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)
