#!/usr/bin/env python3
"""
SCx61 Week 2 — deterministic synthetic test image generator.

Produces the SAME image every time for a given --width/--height (fixed
seed, no downloads, works fully offline). This determinism matters: the serial oracle
(filter_ref.py) and every parallel version (img_mpi_v*.py) must all filter the exact
same pixels to make np.array_equal a meaningful correctness check.

The image is a single-channel (grayscale) uint8 array: a horizontal
gradient, a few overlaid shapes (circles/rectangles at different
intensities), and fixed-seed Gaussian noise. This gives the 3x3 box
blur something visually real to do (gradients blur smoothly, shape
edges blur visibly, noise gets smoothed) and still visible seams if
rows are missing.

Usage:
    python3 make_test_image.py --width 512 --height 512 [--out-prefix test_image]

Writes:
    <out-prefix>.npy   (exact numpy array, used by every script below)
    <out-prefix>.png   (for visual inspection — open it and look)
"""
import argparse
import numpy as np
from PIL import Image

SEED = 42  # fixed — this is what "deterministic" means here


def make_image(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)

    # 1. Horizontal gradient, 0..255 across width, broadcast down height.
    x = np.linspace(0, 255, width, dtype=np.float64)
    gradient = np.tile(x, (height, 1))

    img = gradient.copy()

    # 2. A few fixed shapes at fixed positions (deterministic — no rng
    #    needed for placement, only for noise below).
    yy, xx = np.mgrid[0:height, 0:width]

    # Circle 1: bright disc in the upper-left quadrant.
    cy1, cx1, r1 = height * 0.25, width * 0.25, min(width, height) * 0.12
    circle1 = (xx - cx1) ** 2 + (yy - cy1) ** 2 <= r1 ** 2
    img[circle1] = 230.0

    # Circle 2: dark disc in the lower-right quadrant.
    cy2, cx2, r2 = height * 0.75, width * 0.75, min(width, height) * 0.15
    circle2 = (xx - cx2) ** 2 + (yy - cy2) ** 2 <= r2 ** 2
    img[circle2] = 20.0

    # Rectangle: a mid-tone band across the middle third, full width —
    # this deliberately straddles several strip boundaries regardless
    # of rank count, so strip seams (if any) are visible against it.
    band_top = int(height * 0.45)
    band_bottom = int(height * 0.55)
    img[band_top:band_bottom, :] = 140.0

    # 3. Fixed-seed Gaussian noise, modest sigma so it doesn't swamp
    #    the shapes but is enough to make a 3x3 blur visibly smooth it.
    noise = rng.normal(loc=0.0, scale=8.0, size=(height, width))
    img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out-prefix", type=str, default="test_image")
    args = ap.parse_args()

    img = make_image(args.width, args.height)

    np.save(f"{args.out_prefix}.npy", img)
    Image.fromarray(img, mode="L").save(f"{args.out_prefix}.png")

    print(f"Wrote {args.out_prefix}.npy and {args.out_prefix}.png "
          f"({args.width}x{args.height}, dtype={img.dtype}, seed={SEED})")


if __name__ == "__main__":
    main()
