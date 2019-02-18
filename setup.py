from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


_GCC_WARNING_FLAGS = """
    -pedantic -W -Wall -Wextra -Wcast-align -Wcast-qual
    -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
    -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
    -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
    -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
    -Wstrict-overflow=1 -Wswitch-default
    -Wno-undef -Wno-unused
""".split()


setup(name='nqs_playground',
      version='0.1',
      description='PyTorch-based implementation of SR for NQS',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      url='http://github.com/twesterhout/nqs_playground',
      author='Tom Westerhout',
      author_email='t.westerhout@student.ru.nl',
      license='BSD3',
      packages=['nqs_playground'],
      ext_modules=[
          CppExtension('_C_nqs', sources=['cbits/nqs.cpp'],
              include_dirs=[
                  'cbits',
                  'external',
                  # 'external/pybind11/include',
                  'external/boost/libs/pool/include',
                  ],
              extra_compile_args=["-std=c++14", # "-UNDEBUG", "-O1",
                                  "-msse", "-msse2", "-msse3",
                                  "-msse4", "-msse4.1", "-msse4.2", "-mavx"])
      ],
      cmdclass={
          'build_ext': BuildExtension
      },
      install_requires=[
          'torch>=1.0',
          'numpy>=0.15',
          'scipy',
          'numba',
          'click',
          'mpmath',
          'psutil',
      ],
      zip_safe=False)
