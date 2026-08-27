// Standalone port of MRtrix3's degibbs/unring2d.h.
// Only the Unring2D class is kept - it is entirely self-contained
// (holds its own FFT plans, no dependency on MRtrix's Image/
// ThreadedLoop machinery). The original Unring2DFunctor (which drove
// Unring2D across an MRtrix Image using ThreadedLoop) is dropped; the
// pybind11 bindings loop over slices directly instead.
#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "degibbs_core.h"
#include "fft1d.h"
#include "unring1d.h"

namespace MR::Degibbs {

class Unring2D {
public:
  Unring2D(size_t nrows, size_t ncols, const int nsh, const int minW, const int maxW)
      : row_fft(ncols, FFTW_FORWARD),
        col_fft(nrows, FFTW_FORWARD),
        row_ifft(ncols, FFTW_BACKWARD),
        col_ifft(nrows, FFTW_BACKWARD),
        unring1d_row(row_ifft, nsh, minW, maxW),
        unring1d_col(col_ifft, nsh, minW, maxW),
        slice2(nrows, ncols) {}

  Unring2D(const Unring2D &other)
      : row_fft(other.row_fft.size(), FFTW_FORWARD),
        col_fft(other.col_fft.size(), FFTW_FORWARD),
        row_ifft(other.row_ifft.size(), FFTW_BACKWARD),
        col_ifft(other.col_ifft.size(), FFTW_BACKWARD),
        unring1d_row(row_ifft, other.unring1d_row.nsh, other.unring1d_row.minW, other.unring1d_row.maxW),
        unring1d_col(col_ifft, other.unring1d_col.nsh, other.unring1d_col.minW, other.unring1d_col.maxW),
        slice2(other.slice2.rows(), other.slice2.cols()) {}

  template <typename Derived> inline void operator()(Eigen::MatrixBase<Derived> &slice) {
    assert(slice.cols() == slice2.cols());
    assert(slice.rows() == slice2.rows());

    row_FFT(slice);
    col_FFT(slice);

    for (int k = 0; k < slice.cols(); k++) {
      const real_type ck = (1.0 + cos(2.0 * Math::pi * (real_type(k) / slice.cols()))) * 0.5;
      for (int j = 0; j < slice.rows(); j++) {
        const real_type cj = (1.0 + cos(2.0 * Math::pi * (real_type(j) / slice.rows()))) * 0.5;

        if (ck + cj != 0.0) {
          slice2(j, k) = slice(j, k) * cj / (ck + cj);
          slice(j, k) *= ck / (ck + cj);
        } else
          slice(j, k) = slice2(j, k) = complex_type(0.0, 0.0);
      }
    }

    row_iFFT(slice);
    col_iFFT(slice2);

    for (ssize_t n = 0; n < slice.cols(); ++n)
      unring1d_col(slice.col(n));
    for (ssize_t n = 0; n < slice2.rows(); ++n)
      unring1d_row(slice2.row(n).transpose());

    slice.noalias() = (slice + slice2) / (slice.rows() * slice.cols());
  }

private:
  Math::FFT1D row_fft, col_fft, row_ifft, col_ifft;
  Unring1D unring1d_row, unring1d_col;
  Eigen::Matrix<complex_type, Eigen::Dynamic, Eigen::Dynamic> slice2;

  template <typename fft_obj, typename Derived> inline void FFT(fft_obj &fft, Eigen::MatrixBase<Derived> &M) {
    assert(fft.size() == static_cast<size_t>(M.cols()));
    for (auto n = 0; n < M.rows(); ++n) {
      for (ssize_t i = 0; i < M.cols(); ++i)
        fft[i] = M(n, i);
      fft.run();
      for (ssize_t i = 0; i < M.cols(); ++i)
        M(n, i) = fft[i];
    }
  }
  template <typename Derived> inline void FFT(Math::FFT1D &fft, Eigen::MatrixBase<Derived> &&M) { FFT(fft, M); }

  template <typename Derived> inline void row_FFT(Eigen::MatrixBase<Derived> &mat) { FFT(row_fft, mat); }
  template <typename Derived> inline void row_iFFT(Eigen::MatrixBase<Derived> &mat) { FFT(row_ifft, mat); }
  template <typename Derived> inline void col_FFT(Eigen::MatrixBase<Derived> &mat) { FFT(col_fft, mat.transpose()); }
  template <typename Derived> inline void col_iFFT(Eigen::MatrixBase<Derived> &mat) { FFT(col_ifft, mat.transpose()); }
};

} // namespace MR::Degibbs
