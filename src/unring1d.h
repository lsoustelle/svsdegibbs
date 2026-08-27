// Standalone port of MRtrix3's degibbs/unring1d.h.
// Algorithm logic is unchanged from the original; only the include of
// "math/fft.h" (replaced by our fft1d.h) and the FORCE_INLINE macro
// (replaced by plain inline) were touched.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "degibbs_core.h"
#include "fft1d.h"

namespace MR::Degibbs {

class Unring1D {
public:
  Unring1D(Math::FFT1D &fft, const int nsh, const int minW, const int maxW)
      : nsh(nsh), minW(minW), maxW(maxW), fft(fft), shifted(fft.size(), 2 * nsh + 1), shifts(2 * nsh + 1) {
    shifts[0] = 0;
    for (int j = 0; j < nsh; j++) {
      shifts[j + 1] = j + 1;
      shifts[1 + nsh + j] = -(j + 1);
    }
  }

  template <typename Derived> inline void operator()(Eigen::MatrixBase<Derived> &data) {
    assert(data.cols() == 1);
    assert(fft.size() == static_cast<size_t>(data.size()));

    std::vector<real_type> TV1arr(2 * nsh + 1);
    std::vector<real_type> TV2arr(2 * nsh + 1);

    const int n = fft.size();
    const int maxn = (n & 1) ? (n - 1) / 2 : n / 2 - 1;

    // iFFT original line as-is:
    for (int i = 0; i < n; ++i)
      fft[i] = data[i];
    fft.run();
    for (int i = 0; i < n; ++i)
      shifted(i, 0) = fft[i];

    // apply shifts and iFFT each line:
    for (int j = 1; j < 2 * nsh + 1; j++) {
      const real_type phi = Math::pi * real_type(shifts[j]) / static_cast<real_type>(n * nsh);
      const complex_type u(std::cos(phi), std::sin(phi));
      complex_type e(1.0, 0.0);
      fft[0] = data[0];

      if (!(n & 1))
        fft[n / 2] = complex_type(0.0, 0.0);

      for (int l = 0; l < maxn; l++) {
        e = u * e;
        int L = l + 1;
        fft[L] = e * data[L];
        L = n - 1 - l;
        fft[L] = std::conj(e) * data[L];
      }

      fft.run();
      for (int i = 0; i < n; ++i)
        shifted(i, j) = fft[i];
    }

    for (int j = 0; j < 2 * nsh + 1; ++j) {
      TV1arr[j] = 0.0;
      TV2arr[j] = 0.0;
      for (int t = minW; t <= maxW; t++) {
        TV1arr[j] += std::fabs(shifted((n - t) % n, j).real() - shifted((n - t - 1) % n, j).real());
        TV1arr[j] += std::fabs(shifted((n - t) % n, j).imag() - shifted((n - t - 1) % n, j).imag());
        TV2arr[j] += std::fabs(shifted((n + t) % n, j).real() - shifted((n + t + 1) % n, j).real());
        TV2arr[j] += std::fabs(shifted((n + t) % n, j).imag() - shifted((n + t + 1) % n, j).imag());
      }
    }

    for (int l = 0; l < n; ++l) {
      real_type minTV = std::numeric_limits<real_type>::max();
      int minidx = 0;
      for (int j = 0; j < 2 * nsh + 1; ++j) {

        if (TV1arr[j] < minTV) {
          minTV = TV1arr[j];
          minidx = j;
        }
        if (TV2arr[j] < minTV) {
          minTV = TV2arr[j];
          minidx = j;
        }

        TV1arr[j] += std::fabs(shifted((l - minW + 1 + n) % n, j).real() - shifted((l - (minW) + n) % n, j).real());
        TV1arr[j] -= std::fabs(shifted((l - maxW + n) % n, j).real() - shifted((l - (maxW + 1) + n) % n, j).real());
        TV2arr[j] += std::fabs(shifted((l + maxW + 1 + n) % n, j).real() - shifted((l + (maxW + 2) + n) % n, j).real());
        TV2arr[j] -= std::fabs(shifted((l + minW + n) % n, j).real() - shifted((l + (minW + 1) + n) % n, j).real());

        TV1arr[j] += std::fabs(shifted((l - minW + 1 + n) % n, j).imag() - shifted((l - (minW) + n) % n, j).imag());
        TV1arr[j] -= std::fabs(shifted((l - maxW + n) % n, j).imag() - shifted((l - (maxW + 1) + n) % n, j).imag());
        TV2arr[j] += std::fabs(shifted((l + maxW + 1 + n) % n, j).imag() - shifted((l + (maxW + 2) + n) % n, j).imag());
        TV2arr[j] -= std::fabs(shifted((l + minW + n) % n, j).imag() - shifted((l + (minW + 1) + n) % n, j).imag());
      }

      const real_type a0r = shifted((l - 1 + n) % n, minidx).real();
      const real_type a1r = shifted(l, minidx).real();
      const real_type a2r = shifted((l + 1 + n) % n, minidx).real();
      const real_type a0i = shifted((l - 1 + n) % n, minidx).imag();
      const real_type a1i = shifted(l, minidx).imag();
      const real_type a2i = shifted((l + 1 + n) % n, minidx).imag();
      const real_type s = real_type(shifts[minidx]) / (2.0 * nsh);

      if (s > 0.0)
        data(l) = complex_type(a1r * (1.0 - s) + a0r * s, a1i * (1.0 - s) + a0i * s);
      else
        data(l) = complex_type(a1r * (1.0 + s) - a2r * s, a1i * (1.0 + s) - a2i * s);
    }
  }

  template <typename Derived> inline void operator()(Eigen::MatrixBase<Derived> &&data) { operator()(data); }

  const int nsh, minW, maxW;

private:
  Math::FFT1D &fft;
  Eigen::Matrix<complex_type, Eigen::Dynamic, Eigen::Dynamic> shifted;
  std::vector<int> shifts;
};

} // namespace MR::Degibbs
