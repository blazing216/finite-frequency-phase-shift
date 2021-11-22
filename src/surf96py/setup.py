from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extension = Extension(
    name="surf96py",
    sources=["srfdrr96_cython.pyx"],
    libraries=["srfdis96", "srfdrr96"],
    library_dirs=["/home/xuyh/Autocorrelation/Programs/lib"],
    language='fortran',
)

setup(
    name="srfdis96",
    ext_modules=cythonize([extension],
    compiler_directives={'language_level' : "3"},
    gdb_debug=True)
)
