from setuptools import setup
from sys import version_info

if version_info.major < 3 or version_info.minor < 10:
    raise RuntimeError("amath required 3.10 or new.")

setup(
    name="amath", 
    requires=[
        "sympy",
        "mpmath"
    ],
    author="billloic",
    author_email="billloic6@gmail.com"
)