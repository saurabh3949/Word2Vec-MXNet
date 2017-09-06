from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext_module = cythonize("preprocessing.pyx", annotate = True)
ext_module[0].include_dirs = [numpy.get_include(), '.']


setup(
    ext_modules=ext_module
)