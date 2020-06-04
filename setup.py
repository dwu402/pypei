import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypei",
    version="0.1.0a2",
    author="David Wu",
    author_email="dwu402@aucklanduni.ac.nz",
    description="Implementation of generalised profiling in CasADi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dwu402/pypei",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'casadi>=3.5',
        'numpy>=1.17.4'
    ]
)