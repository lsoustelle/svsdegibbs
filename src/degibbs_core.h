// Standalone port of MRtrix3's degibbs algorithm — core scalar types.
// No dependency on the MRtrix3 core library.
#pragma once

#include <complex>

namespace MR::Degibbs {

using real_type = double;
using complex_type = std::complex<real_type>;

} // namespace MR::Degibbs
