#!/usr/bin/env python3
"""
SCx61 Week 2 — correctness checker against the serial oracle.

Reports the total number of differing pixels between a parallel
result and filter_ref.py's serial output, and WHICH rows differ (this
is what makes the v1 "seam" visible as data, not just as a picture).

Usage:
    python3 compare.py --candidate test_image_filtered_v1
    python3 compare.py --candidate test_image_filtered_v2
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial-prefix", type=str, default="test_image_filtered_serial")
    ap.add_argument("--candidate", type=str, required=True)
    args = ap.parse_args()

    serial = np.load(f"{args.serial_prefix}.npy")
    candidate = np.load(f"{args.candidate}.npy")

    if serial.shape != candidate.shape:
        print(f"SHAPE MISMATCH: serial={serial.shape} candidate={candidate.shape}")
        return

    equal = np.array_equal(serial, candidate)
    diff_mask = serial != candidate
    n_diff = int(diff_mask.sum())

    print(f"Comparing {args.candidate}.npy against {args.serial_prefix}.npy")
    print(f"np.array_equal: {equal}")
    print(f"Differing pixels: {n_diff} / {serial.size}")

    if n_diff > 0:
        diff_rows = sorted(set(np.where(diff_mask)[0].tolist()))
        # Report as contiguous ranges, since a strip-boundary seam is
        # a short run of adjacent rows, not scattered individual rows.
        ranges = []
        start = prev = diff_rows[0]
        for r in diff_rows[1:]:
            if r == prev + 1:
                prev = r
                continue
            ranges.append((start, prev))
            start = prev = r
        ranges.append((start, prev))
        range_str = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)
        print(f"Differing row indices: {range_str}")
    else:
        print("0 differing pixels — exact match with the serial oracle.")


if __name__ == "__main__":
    main()
