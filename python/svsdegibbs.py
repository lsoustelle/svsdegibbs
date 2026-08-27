#!/usr/bin/env python3
"""
svsdegibbs — standalone Python/pybind11 port of MRtrix3's mrdegibbs command.

Named "svsdegibbs" ("SubVoxel Shift" degibbs, after the underlying
Kellner/Bautista subvoxel-shift method) so it's not mistaken for the
original MRtrix3 command when both are installed side by side.

Wraps the compiled `_svsdegibbs` extension (unring2d / unring3d) with
nibabel-based NIfTI I/O, mirroring MRtrix3's mrdegibbs CLI options.
This is a first-pass wrapper meant to be refined further at the Python
level.

Usage:
    svsdegibbs input.nii.gz output.nii.gz [options]

Options mirror MRtrix3's mrdegibbs:
    -dimensionality {2,3}    2 = slice-wise (Kellner et al., default)
                             3 = volume-wise 3D (Bautista et al.)
    -axes list               slice axes for 2D mode (default: 0,1)
    -nshifts N               subpixel discretization (default: 20)
    -minW N                  left TV window border (default: 1)
    -maxW N                  right TV window border (default: 3)
    -nthreads N              number of OpenMP threads (default: 1)

Input is expected to be a 3D (single volume) or 4D ({volume 3D or 2D+multislice} + REP/DWI/etc.
concatenated in the 4th dimension) NIfTI image, real- or complex-valued.
Real input produces real output (imaginary part discarded after
processing, matching MRtrix3's mrdegibbs behaviour); complex input
produces complex output.
"""
import argparse
import math
import os
import sys

import nibabel as nib
import numpy as np
from tqdm import tqdm

try:
    import _svsdegibbs as _core
except ImportError as exc:
    raise ImportError(
        "Could not import the compiled '_svsdegibbs' extension. "
        "Build it first (see README.md) and make sure the resulting "
        "_svsdegibbs*.so is on your PYTHONPATH."
    ) from exc

# tqdm output with just a description and a percentage.
_BAR_FORMAT = "{desc}: {percentage:3.0f}%"


def _parse_axes(s):
    axes = [int(x) for x in s.split(",")]
    if len(axes) != 2:
        raise argparse.ArgumentTypeError("axes must be a comma-separated pair, e.g. 0,1")
    return axes


def build_parser():
    p = argparse.ArgumentParser(description="Remove Gibbs ringing artefacts (standalone port of MRtrix3's mrdegibbs).")
    p.add_argument("input", help="input NIfTI image (3D volume, or 4D volume+REP/DWI stack)")
    p.add_argument("output", help="output NIfTI image")
    p.add_argument("-dimensionality",type=int,choices=[2, 3],default=2,help="2 = slice-wise 2D method (default), 3 = volume-wise 3D method")
    p.add_argument("-axes",type=_parse_axes,default=[0, 1],metavar="list",help="slice axes for 2D mode, comma-separated (default: 0,1)")
    p.add_argument("-nshifts", type=int, default=20, help="discretization of subpixel spacing (default: 20)")
    p.add_argument("-minW", type=int, default=1, help="left border of TV window (default: 1)")
    p.add_argument("-maxW", type=int, default=3, help="right border of TV window (default: 3)")
    p.add_argument("-nthreads", type=int, default=1, metavar="N",help="number of OpenMP threads to use (default: 1; 0 = all available)")
    return p


def _process_in_chunks(core_fn, work, nshifts, minW, maxW, nthreads, desc, chunks_target=40):
    """Run core_fn (unring2d/unring3d) over `work`'s last axis in chunks,
    reporting progress after each chunk. Each chunk is still processed
    with full OpenMP parallelism inside the C++ extension; this only
    trades a bit of that parallelism (chunk_size items at a time instead
    of everything at once) for incremental feedback, since the C++ side
    has no progress callback."""
    total = work.shape[-1]
    chunk_size = max(1, math.ceil(total / chunks_target))
    out = np.empty_like(work)
    with tqdm(total=total, desc=desc, bar_format=_BAR_FORMAT) as pbar:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_in = np.ascontiguousarray(work[..., start:end])
            out[..., start:end] = core_fn(chunk_in, nshifts, minW, maxW, nthreads)
            pbar.update(end - start)
    return out


def run_svsdegibbs(data, dimensionality, slice_axes, nshifts, minW, maxW, nthreads=0):
    """data: real or complex numpy array, ndim 3 (single volume) or 4 (volume x REP)."""
    if minW >= maxW:
        raise ValueError("minW must be smaller than maxW")

    squeeze_rep = False
    if data.ndim == 3:
        data = data[..., np.newaxis]
        squeeze_rep = True
    elif data.ndim != 4:
        raise ValueError(f"expected a 3D or 4D image, got {data.ndim}D")

    work = np.ascontiguousarray(data.astype(np.complex128, copy=False))

    if slice_axes == [0, 1, 2]:
        dimensionality = 3

    if dimensionality > 2:
        out = _process_in_chunks(_core.unring3d, work, nshifts, minW, maxW, nthreads, desc="3D unringing")
    else:
        if len(slice_axes) != 2:
            raise ValueError("slice axes must be a 2-element list")
        if max(slice_axes) >= 3:
            raise ValueError("slice axes must be within the first 3 dimensions of the image")
        if slice_axes[0] == slice_axes[1]:
            raise ValueError("two independent slice axes must be specified")

        outer_axis = ({0, 1, 2} - set(slice_axes)).pop()
        # bring (slice_axis0, slice_axis1, outer_axis) to the front, REP (axis 3) stays last,
        # then flatten (outer_axis, REP) into a single "slices" axis for unring2d.
        perm = (slice_axes[0], slice_axes[1], outer_axis, 3)
        work_t = np.transpose(work, perm)
        n0, n1, n_outer, n_rep = work_t.shape
        flat = np.ascontiguousarray(work_t.reshape(n0, n1, n_outer * n_rep))

        out_flat = _process_in_chunks(_core.unring2d, flat, nshifts, minW, maxW, nthreads, desc="2D unringing")

        out_t = out_flat.reshape(n0, n1, n_outer, n_rep)
        inv_perm = tuple(np.argsort(perm))
        out = np.transpose(out_t, inv_perm)

    if squeeze_rep:
        out = out[..., 0]

    return out


def _resolve_path(path):
    if os.path.isabs(path):
        return os.path.normpath(path)

    pwd = os.environ.get("PWD")
    if pwd and os.path.isdir(pwd):
        return os.path.normpath(os.path.join(pwd, path))

    try:
        return os.path.abspath(path)
    except OSError as exc:
        raise RuntimeError(
            f"Could not resolve '{path}' to an absolute path."
        ) from exc


def main(argv=None):
    args = build_parser().parse_args(argv)

    args.input = _resolve_path(args.input)
    args.output = _resolve_path(args.output)

    img = nib.load(args.input)
    data = np.asanyarray(img.dataobj)
    is_complex_input = np.iscomplexobj(data)

    out = run_svsdegibbs(data, args.dimensionality, args.axes, args.nshifts, args.minW, args.maxW, nthreads=args.nthreads)

    if not is_complex_input:
        out = out.real.astype(np.float32)
    else:
        out = out.astype(data.dtype if np.iscomplexobj(data) else np.complex64)

    out_img = nib.Nifti1Image(out, img.affine, img.header)
    out_img.header.set_data_dtype(out.dtype)
    nib.save(out_img, args.output)

if __name__ == "__main__":
    main()
