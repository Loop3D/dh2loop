import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dh2loop",
    version="0.0.01",
    author="Ranee Joshi & Kavitha Madaiah",
    author_email="ranee.joshi@research.uwa.edu.au",
    description="A package to extract information from drillholes to feed 3D modelling packages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Loop3D/dh2loop",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
