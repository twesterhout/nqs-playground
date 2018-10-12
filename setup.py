from setuptools import setup

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
      install_requires=[
          'numpy',
          'scipy',
          'torch',
          'numba',
          'click'
      ],
      zip_safe=False)
