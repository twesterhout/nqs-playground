from setuptools import setup

setup(
    name="nqs_playground",
    version="0.0.5",
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
    author_email="t.westerhout@student.ru.nl",
    license="BSD3",
    packages=["nqs_playground"],
    # This will break on OSX and Windows...
    package_data={"nqs_playground": ["_C.*.so", "libnqs.so"]},
    install_requires=[],
    zip_safe=False,
)
