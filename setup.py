from setuptools import setup
import os
import re


def get_version(package):
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, package, "__init__.py"), "r") as input:
        result = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', input.read())
    if not result:
        raise ValueError("failed to determine {} version".format(package))
    return result.group(1)


setup(
    name="nqs-playground",
    version=get_version("nqs_playground"),
    description="PyTorch-based implementation of SR and SWO for NQS",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="http://github.com/twesterhout/nqs-playground",
    author="Tom Westerhout",
    author_email="14264576+twesterhout@users.noreply.github.com",
    license="BSD3",
    packages=["nqs_playground"],
    # This will break on OSX and Windows...
    package_data={"nqs_playground": ["_C.*.so", "libnqs.so"]},
    install_requires=[], # "torch", "numpy", "scipy", "loguru"],
    zip_safe=False,
)
