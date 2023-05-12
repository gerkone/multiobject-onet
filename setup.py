try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    "src.utils.libmesh.triangle_hash",
    sources=["src/utils/libmesh/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[numpy_include_dir],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    "src.utils.libmise.mise",
    sources=["src/utils/libmise/mise.pyx"],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    "src.utils.libsimplify.simplify_mesh",
    sources=["src/utils/libsimplify/simplify_mesh.pyx"],
    include_dirs=[numpy_include_dir],
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    "src.utils.libvoxelize.voxelize",
    sources=["src/utils/libvoxelize/voxelize.pyx"],
    libraries=["m"],  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(ext_modules=cythonize(ext_modules), cmdclass={"build_ext": BuildExtension})
