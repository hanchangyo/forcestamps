from distutils.core import setup, Extension

setup(
    name='forcestamp_c',
    version='1.0.0',
    ext_modules=[Extension('forcestamp_c', ['forcestamp.c'])]
)
