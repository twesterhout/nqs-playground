# nqs-playground
PyTorch-based implementation of SR and SWO for NQS.


### Getting started

The suggested way to manage dependencies is by using ``conda``. To compile
the C++ library we need

  * ``cmake`` (>=3.13)
  * ``g++-7`` (or newer); one can try using ``clang``, but it will most
    likely cause all kinds of weird errors because PyTorch binaries are
    compiled with ``g++-4.9``, and one needs to stay binary compatible
    with it.
  * Intel SVML
  * Intel MKL
  * PyTorch 1.2 (or newer)

The only system requirement is ``g++-7``, the rest can be installed using
``conda``. There is an ``environment.yml`` file which specifies the
environment used to develop the code. It contains some extra stuff not
directly needed for using this project. Those packages are marked as
optional and you can comment them out.

Now, compiling the code. 

.. code:: bash

   cd /path/to/nqs-playground
   mkdir build && cd build
   CFLAGS= CXXFLAGS= CC=gcc-7 CXX=g++-7 cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --target install

I prefer to set ``CFLAGS`` and ``CXXFLAGS`` to empty strings, because
``conda`` initialises them to a bunch or options which conflict with my
own. ``cmake`` installs the shared library into ``nqs_playground`` folder.

After that, we can install the Python package with

.. code:: bash

   python3 -m pip install --user .

(consider also adding ``-e`` flag as it allows you to modify files inside
``nqs_playground`` directory without needing to reinstall the package)
