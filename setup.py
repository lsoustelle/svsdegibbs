"""
Build script for the standalone mrdegibbs pybind11 extension.

Intended workflow (conda-forge provides eigen, fftw, pybind11, a
matched C++ compiler, and — on macOS — llvm-openmp; no system-level
package manager or CMake needed):

    conda env create -f environment.yml
    conda activate svsdegibbs
    pip install .

This reads CONDA_PREFIX to find Eigen/FFTW headers and libraries, so it
must be run from an active conda environment. If you'd rather not use
conda (e.g. system-installed libfftw3-dev / libeigen3-dev), the
CMakeLists.txt in this repo still works standalone — see README.md.
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

conda_prefix = os.environ.get("CONDA_PREFIX")
if not conda_prefix:
    raise RuntimeError(
        "CONDA_PREFIX is not set. This build reads Eigen/FFTW from your "
        "active conda environment — run:\n"
        "    conda env create -f environment.yml\n"
        "    conda activate svsdegibbs\n"
        "    pip install .\n"
        "(If you're intentionally not using conda, build via CMakeLists.txt instead.)"
    )

include_dirs = [
    os.path.join(conda_prefix, "include"),
    os.path.join(conda_prefix, "include", "eigen3"),
]
# Windows conda envs use Library\include, Library\lib instead of include/, lib/.
if sys.platform == "win32":
    include_dirs = [
        os.path.join(conda_prefix, "Library", "include"),
        os.path.join(conda_prefix, "Library", "include", "eigen3"),
    ]
    library_dirs = [os.path.join(conda_prefix, "Library", "lib")]
else:
    library_dirs = [os.path.join(conda_prefix, "lib")]

extra_compile_args = ["-O3"] if sys.platform != "win32" else ["/O2"]
extra_link_args = []
runtime_library_dirs = []

if sys.platform == "darwin":
    # conda-forge's clang needs the preprocessor flag form for OpenMP,
    # and links against llvm-openmp's libomp rather than libgomp.
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp", "-L" + library_dirs[0]]
elif sys.platform == "win32":
    extra_compile_args += ["/openmp"]
else:
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]
    runtime_library_dirs = library_dirs  # embed rpath so the built .so finds conda's libfftw3 at import time

ext_modules = [
    Pybind11Extension(
        "_svsdegibbs",
        ["src/bindings.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["fftw3"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        runtime_library_dirs=runtime_library_dirs,
        cxx_std=17,
    ),
]

setup(
    name="svsdegibbs",
    version="0.1.0",
    description="Standalone pybind11 port of MRtrix3's mrdegibbs (Gibbs-ringing removal), named 'svsdegibbs' (SubVoxel Shift degibbs) to avoid clashing with the original command",
    py_modules=["svsdegibbs"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy", "nibabel", "tqdm"],
    entry_points={"console_scripts": ["svsdegibbs=svsdegibbs:main"]},
    zip_safe=False,
)
