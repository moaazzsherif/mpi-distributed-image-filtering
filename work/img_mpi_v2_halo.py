#!/usr/bin/env python3
"""
SCx61 Week 2 — Lab Step 6 (STUDENT STARTER): row-strip decomposition
WITH halo exchange — fill in the TODOs.

Same master-worker distribute/gather as v1. The fix for v1's seams:
before filtering, every rank exchanges its boundary (top/bottom) rows
with its NEIGHBORING ranks — directly rank-to-rank, NOT through rank 0
— using `comm.Sendrecv`, one call that sends your edge row and
receives your neighbor's edge row.

PREDICT: with the real neighboring rows available, how many pixels do
you expect to differ from filter_ref.py's serial oracle now? Check
your prediction with compare.py after you get this running.
"""
import socket
import sys
import numpy as np
from mpi4py import MPI

IN_PREFIX = "test_image"
OUT_PREFIX = "test_image_filtered_v2"


def box_blur_3x3(img: np.ndarray) -> np.ndarray:
    padded = np.pad(img.astype(np.float64), pad_width=1, mode="edge")
    acc = np.zeros_like(img, dtype=np.float64)
    h, w = img.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
    acc /= 9.0
    return np.clip(acc, 0, 255).astype(np.uint8)


def split_rows(height: int, n: int):
    base = height // n
    extra = height % n
    bounds = []
    start = 0
    for i in range(n):
        rows = base + (1 if i < extra else 0)
        bounds.append((start, start + rows))
        start += rows
    return bounds


def halo_exchange(comm, rank, size, strip, width):
    """Returns (top_halo_row, bottom_halo_row), each shape (width,).

    TODO: fill in the two Sendrecv calls below. For each direction you
    are simultaneously a sender (of your own edge row) and a receiver
    (of your neighbor's edge row) — that is exactly what Sendrecv is
    for: one call, both halves.
    """
    top_halo = np.empty(width, dtype=np.uint8)
    bottom_halo = np.empty(width, dtype=np.uint8)

    if rank > 0:
        comm.Sendrecv(
            sendbuf=strip[0, :].copy(), dest=rank - 1, sendtag=30,
            recvbuf=top_halo, source=rank - 1, recvtag=31)
    else:
        # No neighbor above — true top edge of the image: edge-replicate,
        # exactly like filter_ref.py does at row 0.
        top_halo = strip[0, :].copy()

    if rank < size - 1:
        comm.Sendrecv(
            sendbuf=strip[-1, :].copy(), dest=rank + 1, sendtag=31,
            recvbuf=bottom_halo, source=rank + 1, recvtag=30)
    else:
        # True bottom edge of the image — edge-replicate.
        bottom_halo = strip[-1, :].copy()

    return top_halo, bottom_halo


def filter_with_halo(strip: np.ndarray, top_halo: np.ndarray, bottom_halo: np.ndarray) -> np.ndarray:
    extended = np.vstack([top_halo[None, :], strip, bottom_halo[None, :]])
    filtered_extended = box_blur_3x3(extended)
    core_rows = strip.shape[0]
    return filtered_extended[1:1 + core_rows, :]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    in_prefix = sys.argv[1] if len(sys.argv) > 1 else IN_PREFIX
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else OUT_PREFIX

    if rank == 0:
        img = np.load(f"{in_prefix}.npy")
        height, width = img.shape
        bounds = split_rows(height, size)
        print(f"[rank 0 on {hostname}] image {img.shape}, "
              f"split into {size} strips: {bounds}", flush=True)

        my_start, my_end = bounds[0]
        my_strip = img[my_start:my_end, :].copy()

        for worker in range(1, size):
            w_start, w_end = bounds[worker]
            strip = img[w_start:w_end, :].copy()
            comm.send(strip.shape, dest=worker, tag=10)
            comm.Send(strip, dest=worker, tag=11)

        width_for_halo = img.shape[1]
    else:
        shape = comm.recv(source=0, tag=10)
        my_strip = np.empty(shape, dtype=np.uint8)
        comm.Recv(my_strip, source=0, tag=11)
        width_for_halo = shape[1]
        print(f"[rank {rank} on {hostname}] received strip {shape}", flush=True)

    # Every rank (including rank 0) now has its own strip and exchanges
    # halo rows directly with its neighbors.
    top_halo, bottom_halo = halo_exchange(comm, rank, size, my_strip, width_for_halo)
    my_filtered = filter_with_halo(my_strip, top_halo, bottom_halo)

    if rank == 0:
        result = np.zeros_like(img)
        result[my_start:my_end, :] = my_filtered
        for worker in range(1, size):
            w_start, w_end = bounds[worker]
            shape = comm.recv(source=worker, tag=20)
            buf = np.empty(shape, dtype=np.uint8)
            comm.Recv(buf, source=worker, tag=21)
            result[w_start:w_end, :] = buf

        np.save(f"{out_prefix}.npy", result)
        print(f"[rank 0 on {hostname}] wrote {out_prefix}.npy "
              f"(WITH halo exchange — expect 0 differing pixels vs serial)", flush=True)
    else:
        comm.send(my_filtered.shape, dest=0, tag=20)
        comm.Send(my_filtered, dest=0, tag=21)


if __name__ == "__main__":
    main()
