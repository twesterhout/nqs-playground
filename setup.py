import glob
from setuptools import setup

setup(name='nqs_playground',
      version='0.0.1',
      description='PyTorch-based implementation of SR and SWO for NQS',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      url='http://github.com/twesterhout/nqs-playground',
      author='Tom Westerhout',
      author_email='t.westerhout@student.ru.nl',
      license='BSD3',
      packages=['nqs_playground'],
      package_data={'nqs_playground': ['_C_nqs.*']},
      install_requires=[
          # 'numpy',
          # 'scipy',
          # 'torch>=1.2',
          # 'numba',
          # 'tqdm',
          # 'pytest',
          # 'pytorch-ignite',
          # 'mpi4py',
      ],
      zip_safe=False)
