# svsdegibbs - standalone pybind11 port of MRtrix3's mrdegibbs

A standalone C++/pybind11 port of MRtrix3's `mrdegibbs` (Gibbs-ringing removal, Kellner et al. 2D / Bautista et al. 3D), with no dependency on the MRtrix3 core library.
Codes are derived from [mrtrix3 dev branch](https://github.com/MRtrix3/mrtrix3/tree/dev) (hash `#76556c6`). 
Only Eigen3, FFTW3, pybind11, and tqdm are needed. 
Deliberately renamed `svsdegibbs` ("SubVoxel Shift" degibbs, after the underlying Kellner method), not `mrdegibbs`, so it isn't mistaken for the original MRtrix3 command when both are installed side by side.

## Layout

```
src/fft1d.h           FFTW3-backed drop-in for MRtrix's Math::FFT1D + Math::pi
src/unring1d.h        1D subvoxel-shift core
src/unring2d.h        2D slice-wise method
src/unring3d.h        3D volume-wise method 
src/bindings.cpp      pybind11 module: unring2d(), unring3d()
python/svsdegibbs.py  nibabel-based CLI wrapper mirroring MRtrix3's
                      mrdegibbs options
```

## Installation

Only a conda environment is required. 
`environment.yml` pulls Eigen, FFTW3, pybind11, a matched C++ compiler (`cxx-compiler`), and (for macOS) `llvm-openmp` from conda-forge:

```bash
conda env create -f environment.yml
conda activate svsdegibbs
pip install .
```

## Usage
See `svsdegibbs --help`
### 3D case
Note that `-axes` option is omitted (irrelevant in 3D). 
```bash
svsdegibbs input.nii.gz output.nii.gz -dimensionality 3 -nthreads 6
```

### 2D case
```bash
svsdegibbs input.nii.gz output.nii.gz -dimensionality 2 -axes 0,1 -nshifts 20 -minW 1 -maxW 3 -nthreads 0
```

## References
- Kellner, E; Dhital, B; Kiselev, V.G & Reisert, M. Gibbs-ringing artifact removal based on local subvoxel-shifts. Magnetic Resonance in Medicine, 2016, 76, 1574-1581.

- Bautista, T; O’Muircheartaigh, J; Hajnal, JV; & Tournier, J-D. Removal of Gibbs ringing artefacts for 3D acquisitions using subvoxel shifts. Proc. ISMRM, 2021, 29, 3535.

- Tournier, J.-D.; Smith, R. E.; Raffelt, D.; Tabbara, R.; Dhollander, T.; Pietsch, M.; Christiaens, D.; Jeurissen, B.; Yeh, C.-H. & Connelly, A. MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. NeuroImage, 2019, 202, 116137