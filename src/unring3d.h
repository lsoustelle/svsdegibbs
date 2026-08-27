// Standalone reimplementation of MRtrix3's degibbs/unring3d.h.
// The algorithm (Bautista et al. 3D extension of Kellner's subvoxel
// shift method) is unchanged. What's rewritten is only the plumbing:
// the original drove everything through MRtrix's Image<T>, Header,
// ThreadedLoop and Loop/Iterator classes. Here we operate directly on
// a flat complex buffer of shape (nx, ny, nz), C-contiguous with nz
// fastest-varying (i.e. the same layout numpy gives us by default).

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "degibbs_core.h"
#include "fft1d.h"

namespace MR::Degibbs {

namespace detail3d {

inline real_type indexshift(int n, int size) { return real_type(n > size / 2 ? n - size : n); }

inline int wraparound(int n, int size) { return ((n % size) + size) % size; }

struct LineView {
  complex_type *base;
  size_t stride;
  complex_type &operator[](int i) { return base[static_cast<size_t>(i) * stride]; }
};

inline int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int this_thread() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

} // namespace detail3d

// Processes successive (nx,ny,nz) complex volumes.
class Unring3DVolumeProcessor {
public:
  Unring3DVolumeProcessor(int nx, int ny, int nz, int minW, int maxW, int num_shifts, int nthreads = 0)
      : nx(nx), ny(ny), nz(nz), minW(minW), maxW(maxW), num_shifts(num_shifts),
        nthreads_(nthreads > 0 ? nthreads : detail3d::max_threads()),
        vol_FT(static_cast<size_t>(nx) * ny * nz), vol_filtered(static_cast<size_t>(nx) * ny * nz) {
    const int nthreads_pool = nthreads_;

    shift_frac.resize(2 * num_shifts + 1);
    shift_frac[0] = 0.0;
    for (int j = 0; j < num_shifts; ++j) {
      shift_frac[j + 1] = (j + 1) / (2.0 * num_shifts + 1.0);
      shift_frac[1 + num_shifts + j] = -shift_frac[j + 1];
    }

    for (int axis = 0; axis < 3; ++axis) {
      const int dim = size_of(axis);
      fft_pool[axis].reserve(nthreads_pool);
      ifft_pool[axis].resize(nthreads_pool);
      for (int t = 0; t < nthreads_pool; ++t) {
        fft_pool[axis].emplace_back(dim, FFTW_FORWARD);
        ifft_pool[axis][t].reserve(2 * num_shifts + 1);
        for (int f = 0; f < 2 * num_shifts + 1; ++f)
          ifft_pool[axis][t].emplace_back(dim, FFTW_BACKWARD);
      }
    }

    {
      std::lock_guard<std::mutex> lock(fftw_planner_mutex());
      plan_fwd3d = fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(vol_FT.data()),
                                     reinterpret_cast<fftw_complex *>(vol_FT.data()), FFTW_FORWARD, FFTW_ESTIMATE);
      plan_bwd3d =
          fftw_plan_dft_3d(nx, ny, nz, reinterpret_cast<fftw_complex *>(vol_filtered.data()),
                            reinterpret_cast<fftw_complex *>(vol_filtered.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  ~Unring3DVolumeProcessor() {
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    fftw_destroy_plan(plan_fwd3d);
    fftw_destroy_plan(plan_bwd3d);
  }

  Unring3DVolumeProcessor(const Unring3DVolumeProcessor &) = delete;
  Unring3DVolumeProcessor &operator=(const Unring3DVolumeProcessor &) = delete;

  void process(const complex_type *in, complex_type *out) {
    const size_t N = static_cast<size_t>(nx) * ny * nz;

    std::copy(in, in + N, vol_FT.data());
    fftw_execute(plan_fwd3d); // vol_FT: unnormalized forward 3D transform

    std::fill(out, out + N, complex_type(0.0, 0.0));

    for (int axis = 0; axis < 3; ++axis) {
      apply_filter(axis);
      fftw_execute(plan_bwd3d); // vol_filtered: unnormalized backward 3D transform, in place
      process_lines(axis, out);
    }
  }

private:
  int nx, ny, nz, minW, maxW, num_shifts;
  int nthreads_;
  std::vector<complex_type> vol_FT, vol_filtered;
  std::vector<Math::FFT1D> fft_pool[3];               // [axis][thread]
  std::vector<std::vector<Math::FFT1D>> ifft_pool[3]; // [axis][thread][shift]
  std::vector<real_type> shift_frac;
  fftw_plan plan_fwd3d, plan_bwd3d;

  inline int size_of(int axis) const { return axis == 0 ? nx : (axis == 1 ? ny : nz); }
  inline size_t idx(int i, int j, int k) const {
    return (static_cast<size_t>(i) * ny + j) * static_cast<size_t>(nz) + k;
  }

  void apply_filter(int axis) {
    using detail3d::indexshift;
    for (int i = 0; i < nx; ++i) {
      const real_type xi = 1.0 + std::cos(2.0 * Math::pi * indexshift(i, nx) / real_type(nx));
      for (int j = 0; j < ny; ++j) {
        const real_type xj = 1.0 + std::cos(2.0 * Math::pi * indexshift(j, ny) / real_type(ny));
        for (int k = 0; k < nz; ++k) {
          const real_type xk = 1.0 + std::cos(2.0 * Math::pi * indexshift(k, nz) / real_type(nz));
          const real_type w0 = xj * xk, w1 = xi * xk, w2 = xi * xj;
          const real_type denom = w0 + w1 + w2;
          const real_type w = (axis == 0) ? w0 : (axis == 1 ? w1 : w2);
          const size_t o = idx(i, j, k);
          vol_filtered[o] = vol_FT[o] * (denom == real_type(0) ? real_type(0) : (w / denom));
        }
      }
    }
  }

  // For every line along `axis`: forward-FFT the (already spatial-
  // domain, filtered) line, build 2*num_shifts+1 subvoxel-shifted
  // versions via a frequency-domain phase ramp + inverse FFT, pick the
  // shift that minimizes local total variation at each sample, and
  // accumulate the interpolated value into `out`. Parallelized over
  // lines: each thread uses its own pre-built FFT1D plans from the pool
  // (built once in the constructor), so no planning happens here.
  void process_lines(int axis, complex_type *out) {
    using detail3d::indexshift;
    using detail3d::LineView;
    using detail3d::wraparound;

    const int lsize = size_of(axis);
    const real_type scale = real_type(nx) * real_type(ny) * real_type(nz) * real_type(lsize);

    // Enumerate lines along `axis` as (base_index, stride) pairs.
    std::vector<size_t> line_base;
    size_t stride;
    if (axis == 0) {
      stride = static_cast<size_t>(ny) * nz;
      line_base.reserve(static_cast<size_t>(ny) * nz);
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
          line_base.push_back(idx(0, j, k));
    } else if (axis == 1) {
      stride = static_cast<size_t>(nz);
      line_base.reserve(static_cast<size_t>(nx) * nz);
      for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
          line_base.push_back(idx(i, 0, k));
    } else {
      stride = 1;
      line_base.reserve(static_cast<size_t>(nx) * ny);
      for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
          line_base.push_back(idx(i, j, 0));
    }

    const int nlines = static_cast<int>(line_base.size());
    complex_type *vol_filtered_ptr = vol_filtered.data();

#pragma omp parallel num_threads(nthreads_)
    {
      const int tid = detail3d::this_thread();
      Math::FFT1D &fft = fft_pool[axis][tid];
      std::vector<Math::FFT1D> &ifft = ifft_pool[axis][tid];
      const int nshifts_total = static_cast<int>(ifft.size());

      auto run_line = [&](LineView filtered_line, LineView out_line) {
        for (int n = 0; n < lsize; ++n)
          fft[n] = filtered_line[n];
        fft.run();

        for (int f = 0; f < nshifts_total; ++f) {
          for (int n = 0; n < lsize; ++n)
            ifft[f][n] =
                fft[n] * std::exp(complex_type(0.0, 1.0) * 2.0 * indexshift(n, lsize) * Math::pi * shift_frac[f] /
                                   real_type(lsize));
          if (!(lsize & 1))
            ifft[f][lsize / 2] = real_type(0);
          ifft[f].run();
        }

        for (int n = 0; n < lsize; ++n) {
          int best = 0;
          real_type best_var = std::numeric_limits<real_type>::max();
          for (int f = 0; f < nshifts_total; ++f) {
            real_type sum_left = 0.0, sum_right = 0.0;
            for (int k = minW; k <= maxW; ++k) {
              sum_left += std::fabs(ifft[f][wraparound(n - k, lsize)].real() - ifft[f][wraparound(n - k - 1, lsize)].real());
              sum_left += std::fabs(ifft[f][wraparound(n - k, lsize)].imag() - ifft[f][wraparound(n - k - 1, lsize)].imag());
              sum_right += std::fabs(ifft[f][wraparound(n + k, lsize)].real() - ifft[f][wraparound(n + k + 1, lsize)].real());
              sum_right += std::fabs(ifft[f][wraparound(n + k, lsize)].imag() - ifft[f][wraparound(n + k + 1, lsize)].imag());
            }
            const real_type tot = std::min(sum_left, sum_right);
            if (tot < best_var) {
              best_var = tot;
              best = f;
            }
          }

          const real_type shift = shift_frac[best];
          const complex_type a0 = ifft[best][wraparound(n - 1, lsize)];
          const complex_type a1 = ifft[best][n];
          const complex_type a2 = ifft[best][wraparound(n + 1, lsize)];

          if (shift > 0.0)
            out_line[n] += (a1 - shift * (a1 - a0)) / scale;
          else
            out_line[n] += (a1 + shift * (a1 - a2)) / scale;
        }
      };

#pragma omp for schedule(dynamic)
      for (int li = 0; li < nlines; ++li) {
        const size_t base = line_base[li];
        run_line({vol_filtered_ptr + base, stride}, {out + base, stride});
      }
    }
  }
};

} // namespace MR::Degibbs
