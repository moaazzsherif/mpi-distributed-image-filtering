#!/usr/bin/env python3
"""
SCx61 Week 2 — serial reference filter. THE ORACLE.

This is the one piece of ground truth every parallel version
(img_mpi_v1/v2/v3) is checked against via compare.py. If a parallel
result doesn't match this, byte for byte, the parallel code is wrong —
full stop, regardless of how plausible its own output looks.

Filter: 3x3 box blur (mean of the 3x3 neighborhood around each pixel),
edge-replicated at the image boundary (so the image doesn't shrink and
there's no "what do I do at row 0" ambiguity — same edge handling is
used by every MPI version below, applied to their OWN local view of
the image, which is exactly why a strip missing its true neighbor rows
produces a DIFFERENT result at the strip boundary: it edge-replicates
there too, but edge-replicating a strip boundary is not the same as
having the real neighboring pixels).

Usage:
    python3 filter_ref.py [--in-prefix test_image] [--out-prefix test_image_filtered_serial]
"""
import argparse
import numpy as np


def box_blur_3x3(img: np.ndarray) -> np.ndarray:
    """3x3 box blur via 9 shifted, edge-padded views — no explicit
    Python loop over pixels (fast, and the exact same arithmetic every
    version in this lab uses, so results are bit-for-bit comparable)."""
    padded = np.pad(img.astype(np.float64), pad_width=1, mode="edge")
    acc = np.zeros_like(img, dtype=np.float64)
    h, w = img.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
    acc /= 9.0
    return np.clip(acc, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-prefix", type=str, default="test_image")
    ap.add_argument("--out-prefix", type=str, default="test_image_filtered_serial")
    args = ap.parse_args()

    img = np.load(f"{args.in_prefix}.npy")
    filtered = box_blur_3x3(img)
    np.save(f"{args.out_prefix}.npy", filtered)

    print(f"Serial reference filter complete: {img.shape} -> "
          f"{args.out_prefix}.npy (this is the oracle every MPI version "
          f"is checked against)")


if __name__ == "__main__":
    main()
