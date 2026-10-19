#!/usr/bin/env python3
"""
SCx61 Week 2 — Lab Step 5 (STUDENT STARTER): row-strip decomposition,
NO halo exchange — fill in the TODOs.

Master-worker pattern: rank 0 splits the image into `size` contiguous
row strips, sends each worker its own strip, each rank filters ONLY
its own strip in isolation, and rank 0 gathers the results back into a
full image.

THIS VERSION HAS NO HALO EXCHANGE ON PURPOSE (that's Step 6). Predict
before you run: every rank's 3x3 box blur edge-replicates at the TOP
and BOTTOM of its own strip — for the TRUE top/bottom of the image
that's correct, but for an INTERNAL strip boundary it's wrong, because
the real neighboring pixels live in a different rank's strip.

PREDICT: how many rows of the final image do you expect to differ from
filter_ref.py's serial output, and WHERE? Write your prediction down
before running — then check it with compare.py.
"""
import socket
import numpy as np
from mpi4py import MPI

IN_PREFIX = "test_image"
OUT_PREFIX = "test_image_filtered_v1"


def box_blur_3x3(img: np.ndarray) -> np.ndarray:
    """Same arithmetic as filter_ref.py's oracle — the ONLY thing that
    should differ is what each rank can see, not how it computes."""
    padded = np.pad(img.astype(np.float64), pad_width=1, mode="edge")
    acc = np.zeros_like(img, dtype=np.float64)
    h, w = img.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
    acc /= 9.0
    return np.clip(acc, 0, 255).astype(np.uint8)


def split_rows(height: int, n: int):
    """Row ranges for n strips covering [0, height), as even as
    possible. Returns a list of (start, end) exclusive-end pairs,
    len == n, covering every row exactly once."""
    base = height // n
    extra = height % n
    bounds = []
    start = 0
    for i in range(n):
        rows = base + (1 if i < extra else 0)
        bounds.append((start, start + rows))
        start += rows
    return bounds


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    if rank == 0:
        img = np.load(f"{IN_PREFIX}.npy")
        height, width = img.shape
        bounds = split_rows(height, size)
        print(f"[rank 0 on {hostname}] image {img.shape}, "
              f"split into {size} strips: {bounds}", flush=True)

        my_start, my_end = bounds[0]
        my_strip = img[my_start:my_end, :].copy()

        # TODO: for every worker rank (1 .. size-1):
        #   1. slice out that worker's row range from `img` using `bounds[worker]`
        #   2. send the strip's SHAPE (a small Python tuple) with comm.send
        #      (lowercase — pickle-based, fine for small metadata), tag=10
        #   3. send the strip's DATA with comm.Send (uppercase — buffer-based,
        #      for the real NumPy pixel array), tag=11
        for worker in range(1, size):
            w_start, w_end = bounds[worker]
            strip = img[w_start:w_end, :].copy()
            comm.send(strip.shape, dest=worker, tag=10)
            comm.Send(strip, dest=worker, tag=11)

        my_filtered = box_blur_3x3(my_strip)

        # Gather results back.
        result = np.zeros_like(img)
        result[my_start:my_end, :] = my_filtered
        for worker in range(1, size):
            w_start, w_end = bounds[worker]
            shape = comm.recv(source=worker, tag=20)
            buf = np.empty(shape, dtype=np.uint8)
            comm.Recv(buf, source=worker, tag=21)
            result[w_start:w_end, :] = buf

        np.save(f"{OUT_PREFIX}.npy", result)
        print(f"[rank 0 on {hostname}] wrote {OUT_PREFIX}.npy "
              f"(NO halo — expect seams at strip boundaries)", flush=True)

    else:
        # TODO: receive your strip from rank 0, mirroring what rank 0 sent:
        #   1. shape = comm.recv(source=0, tag=10)
        #   2. allocate an empty uint8 buffer of that shape
        #   3. comm.Recv into it, source=0, tag=11
        shape = comm.recv(source=0, tag=10)
        strip = np.empty(shape, dtype=np.uint8)
        comm.Recv(strip, source=0, tag=11)

        print(f"[rank {rank} on {hostname}] received strip {shape}, filtering alone (no halo)", flush=True)

        filtered = box_blur_3x3(strip)

        comm.send(filtered.shape, dest=0, tag=20)
        comm.Send(filtered, dest=0, tag=21)


if __name__ == "__main__":
    main()
