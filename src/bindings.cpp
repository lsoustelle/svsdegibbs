#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "degibbs_core.h"
#include "unring2d.h"
#include "unring3d.h"

namespace py = pybind11;
using MR::Degibbs::complex_type;

namespace {

// nthreads<=0 means "use whatever OpenMP would use by default" (its own
// max_threads / OMP_NUM_THREADS). A positive value pins the thread count
// explicitly for this call only, without touching global OpenMP state.
inline int resolve_nthreads(int nthreads) {
  if (nthreads > 0)
    return nthreads;
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

} // namespace

// input/output: complex128 array, shape (nrows, ncols, nslices).
// Every (nrows, ncols) slice along the 3rd axis is unrung
// independently (2D, slice-wise method — Kellner et al.), slices
// processed in parallel over OpenMP threads.
static py::array_t<std::complex<double>> unring2d(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> input,
                                                    int nshifts, int minW, int maxW, int nthreads) {
  auto buf = input.request();
  if (buf.ndim != 3)
    throw std::runtime_error("unring2d expects a 3D array (nrows, ncols, nslices)");
  if (minW >= maxW)
    throw std::runtime_error("minW must be smaller than maxW");

  const int nrows = static_cast<int>(buf.shape[0]);
  const int ncols = static_cast<int>(buf.shape[1]);
  const int nslices = static_cast<int>(buf.shape[2]);
  const int nthreads_resolved = resolve_nthreads(nthreads);

  py::array_t<std::complex<double>> result({nrows, ncols, nslices});
  auto rbuf = result.request();
  const complex_type *in_ptr = static_cast<const complex_type *>(buf.ptr);
  complex_type *out_ptr = static_cast<complex_type *>(rbuf.ptr);

  MR::Degibbs::Unring2D template_processor(nrows, ncols, nshifts, minW, maxW);

  {
    py::gil_scoped_release release;
#pragma omp parallel num_threads(nthreads_resolved)
    {
      MR::Degibbs::Unring2D local(template_processor);
      Eigen::Matrix<complex_type, Eigen::Dynamic, Eigen::Dynamic> slice(nrows, ncols);

#pragma omp for schedule(dynamic)
      for (int s = 0; s < nslices; ++s) {
        for (int r = 0; r < nrows; ++r)
          for (int c = 0; c < ncols; ++c)
            slice(r, c) = in_ptr[(static_cast<size_t>(r) * ncols + c) * nslices + s];

        local(slice);

        for (int r = 0; r < nrows; ++r)
          for (int c = 0; c < ncols; ++c)
            out_ptr[(static_cast<size_t>(r) * ncols + c) * nslices + s] = slice(r, c);
      }
    }
  }

  return result;
}

// input/output: complex128 array, shape (nx, ny, nz, nvol).
// Every (nx, ny, nz) volume along the 4th axis is unrung independently
// (3D, volume-wise method — Bautista et al.). Volumes are processed
// sequentially, one Unring3DVolumeProcessor instance for the whole
// call — but that processor parallelizes internally at the *line*
// level (see unring3d.h), so even a single lone volume (no REP) keeps
// every core busy, rather than requiring multiple volumes to hand out
// to threads.
static py::array_t<std::complex<double>> unring3d(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> input,
                                                    int nshifts, int minW, int maxW, int nthreads) {
  auto buf = input.request();
  if (buf.ndim != 4)
    throw std::runtime_error("unring3d expects a 4D array (nx, ny, nz, nvol)");
  if (minW >= maxW)
    throw std::runtime_error("minW must be smaller than maxW");

  const int nx = static_cast<int>(buf.shape[0]);
  const int ny = static_cast<int>(buf.shape[1]);
  const int nz = static_cast<int>(buf.shape[2]);
  const int nvol = static_cast<int>(buf.shape[3]);
  const int nthreads_resolved = resolve_nthreads(nthreads);

  py::array_t<std::complex<double>> result({nx, ny, nz, nvol});
  auto rbuf = result.request();
  const complex_type *in_ptr = static_cast<const complex_type *>(buf.ptr);
  complex_type *out_ptr = static_cast<complex_type *>(rbuf.ptr);

  {
    py::gil_scoped_release release;

    MR::Degibbs::Unring3DVolumeProcessor processor(nx, ny, nz, minW, maxW, nshifts, nthreads_resolved);
    std::vector<complex_type> vol_in(static_cast<size_t>(nx) * ny * nz);
    std::vector<complex_type> vol_out(static_cast<size_t>(nx) * ny * nz);

    for (int v = 0; v < nvol; ++v) {
#pragma omp parallel for num_threads(nthreads_resolved) schedule(static)
      for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
          for (int k = 0; k < nz; ++k)
            vol_in[(static_cast<size_t>(i) * ny + j) * nz + k] =
                in_ptr[((static_cast<size_t>(i) * ny + j) * nz + k) * nvol + v];

      processor.process(vol_in.data(), vol_out.data());

#pragma omp parallel for num_threads(nthreads_resolved) schedule(static)
      for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
          for (int k = 0; k < nz; ++k)
            out_ptr[((static_cast<size_t>(i) * ny + j) * nz + k) * nvol + v] =
                vol_out[(static_cast<size_t>(i) * ny + j) * nz + k];
    }
  }

  return result;
}

PYBIND11_MODULE(_svsdegibbs, m) {
  m.doc() = "Standalone pybind11 port of MRtrix3's mrdegibbs (Kellner et al. / Bautista et al.)";
  m.def("unring2d", &unring2d, py::arg("input"), py::arg("nshifts") = 20, py::arg("minW") = 1, py::arg("maxW") = 3,
        py::arg("nthreads") = 1,
        "2D slice-wise Gibbs ringing removal. input: complex128 array (nrows, ncols, nslices).");
  m.def("unring3d", &unring3d, py::arg("input"), py::arg("nshifts") = 20, py::arg("minW") = 1, py::arg("maxW") = 3,
        py::arg("nthreads") = 1,
        "3D volume-wise Gibbs ringing removal. input: complex128 array (nx, ny, nz, nvol).");
}
