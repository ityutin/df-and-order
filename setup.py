import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="df-and-order",
    version="0.0.1",
    description="Read and transform your datasets in convenient and predictable manner.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ityutin/df-and-order",
    author="Ilya Tyutin",
    author_email="emmarrgghh@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["pandas"],
)
