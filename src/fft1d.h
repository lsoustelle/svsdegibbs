// Standalone replacement for MRtrix3's math/fft.h — just the 1D
// complex-to-complex piece that unring1d.h / unring2d.h actually use.
//
// Semantics (inferred from how MRtrix3 uses Math::FFT1D in the degibbs
// code): a thin wrapper around a single fftw_plan_dft_1d of fixed size
// and fixed direction (FFTW_FORWARD / FFTW_BACKWARD), operating
// in-place on an internal buffer accessed via operator[], unnormalized
// (the 1/N scaling is applied explicitly by the calling code, exactly
// as raw FFTW does it). This matches standard FFTW frequency ordering
// (DC at index 0, positive freqs then negative freqs), which is also
// what the indexshift() helpers throughout this codebase assume.
#pragma once

#include <fftw3.h>
#include <mutex>
#include <vector>

#include "degibbs_core.h"

namespace MR::Degibbs {

// FFTW plan creation/destruction is not thread-safe by default; guard
// it globally so we can safely build per-thread FFT1D instances under
// OpenMP.
inline std::mutex &fftw_planner_mutex() {
  static std::mutex m;
  return m;
}

} // namespace MR::Degibbs

namespace Math {

class FFT1D {
public:
  FFT1D(size_t n, int sign) : n_(n), data_(n), plan_(nullptr), sign_(sign) {
    std::lock_guard<std::mutex> lock(MR::Degibbs::fftw_planner_mutex());
    plan_ = fftw_plan_dft_1d(static_cast<int>(n_), reinterpret_cast<fftw_complex *>(data_.data()),
                              reinterpret_cast<fftw_complex *>(data_.data()), sign, FFTW_ESTIMATE);
  }

  FFT1D(const FFT1D &other) : n_(other.n_), data_(other.n_), plan_(nullptr) {
    std::lock_guard<std::mutex> lock(MR::Degibbs::fftw_planner_mutex());
    plan_ = fftw_plan_dft_1d(static_cast<int>(n_), reinterpret_cast<fftw_complex *>(data_.data()),
                              reinterpret_cast<fftw_complex *>(data_.data()), other.sign_, FFTW_ESTIMATE);
    sign_ = other.sign_;
  }

  FFT1D &operator=(const FFT1D &) = delete;

  FFT1D(FFT1D &&other) noexcept : n_(other.n_), data_(std::move(other.data_)), plan_(other.plan_), sign_(other.sign_) {
    other.plan_ = nullptr;
  }

  ~FFT1D() {
    if (plan_) {
      std::lock_guard<std::mutex> lock(MR::Degibbs::fftw_planner_mutex());
      fftw_destroy_plan(plan_);
    }
  }

  size_t size() const { return n_; }
  MR::Degibbs::complex_type &operator[](size_t i) { return data_[i]; }
  const MR::Degibbs::complex_type &operator[](size_t i) const { return data_[i]; }
  void run() { fftw_execute(plan_); }

private:
  size_t n_;
  std::vector<MR::Degibbs::complex_type> data_;
  fftw_plan plan_;
  int sign_ = FFTW_FORWARD;
};

constexpr double pi = 3.14159265358979323846;

} // namespace Math
