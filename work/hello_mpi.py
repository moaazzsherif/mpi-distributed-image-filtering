#!/usr/bin/env python3
"""
SCx61 Week 2 — Lab Step 3: "Hello from 6 Ranks" (STUDENT STARTER)

PREDICT FIRST: before you fill in a single
TODO, write down on paper what you expect to see when this runs as
`mpirun --hostfile ~/work/hostfile -n 6 python3 ~/work/hello_mpi.py`.
How many lines of output? Will every line be identical? Which physical
container will each rank actually run on?

This is SPMD (Single Program, Multiple Data): the exact same script
runs in all 6 processes at once. There is no "master copy" and "worker
copies" of this file — it's one file, six simultaneous executions, and
each one only knows its own `rank` unless it asks the communicator.
"""
import socket
from mpi4py import MPI

# TODO 1: get the communicator every MPI program starts with. This is
# the group of all processes launched together — mpi4py exposes it
# directly, you don't construct it yourself.
comm = MPI.COMM_WORLD

# TODO 2: get THIS process's own rank within that communicator — an
# integer from 0 to (size - 1), different in every one of the 6
# processes even though they're all running this exact same file.
rank = comm.Get_rank()

# TODO 3: get the total number of processes in that communicator (the
# "6" in "-n 6"). Every rank sees the same value here.
size = comm.Get_size()

# This part is done for you: which container (Docker hostname) is THIS
# process physically running on. This is the part that proves mpirun
# really did launch processes across node0/node1/node2 over SSH, not
# just locally.
hostname = socket.gethostname()

print(f"rank {rank} of {size} on {hostname}", flush=True)
