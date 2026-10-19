# Incident Report: Concurrency Deadlock in MPI Halo Exchange Under Data Scaling

**Author:** Moaaz Sherif  
**Classification:** Concurrency Deadlock / Cyclic Blocking Communication  
**Impact:** Total execution freeze under image scaling ($\ge 8192$ px)  

---

## 1. Summary

During performance testing of a parallel stencil computation across a 3-node Open MPI cluster, scaling the input image from width 256 to 8192 caused all 6 MPI ranks to freeze indefinitely during halo exchange. The root cause was an unsafe sequence of blocking `comm.Send` calls that induced a circular wait once message payloads exceeded the MPI eager buffer threshold, forcing synchronous rendezvous semantics.

---

## 2. Symptoms

- **Command Executed:** 
  ```bash
  timeout 20 mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py break_8192 break_8192_out_v3
  ```
- **Observed Behavior:** Each rank printed `[rank N on nodeX] before Send`. The program subsequently hung with zero CPU activity, zero error output, and was terminated after 20 seconds by Linux `timeout` with exit code `124`.
- **Infrastructure Status:** A concurrent `docker compose ps` confirmed all containers remained in an active `Up (healthy)` state. Container orchestration and healthchecks were blind to the deadlock because application processes were not crashing; they were suspended in kernel wait states.

---

## 3. Diagnostic Evidence

### Command Outputs

1. **Cluster Liveness Verification:**
   ```bash
   $ mpirun --hostfile hostfile -n 6 hostname
   node0
   node0
   node1
   node1
   node2
   node2
   ```
   *Verdict:* Proved SSH transport, DNS resolution, and process launching were completely functional across all nodes.

2. **Rank Progression:**
   Inspection of stdout confirmed that all 6 ranks reached the barrier prior to halo communication (`before Send`), but none reached the post-send marker (`after Send`).

3. **Size Sensitivity & Synchronous Probe:**
   - Small image ($256 \times 300$): Ran to completion with 0 errors.
   - Large image ($8192 \times 300$): Hung indefinitely.
   - Small image with `comm.Ssend` (Synchronous Send):
     ```bash
     timeout 20 mpirun --hostfile hostfile -n 6 python3 v3_ssend_test.py small256 small256_out
     rc=124
     ```
     *Verdict:* Forcing synchronous mode on the small image reproduced the hang immediately. This proved the naive algorithm was flawed even on small inputs, succeeding solely by virtue of incidental MPI eager buffering.

---

## 4. Root Cause Analysis

The vulnerability resided in the naive halo exchange loop:

```python
# img_mpi_v3_naive_halo.py (lines 93-96)
if rank > 0:
    comm.Send(my_strip[0, :].copy(), dest=rank - 1, tag=TAG_UP)
if rank < size - 1:
    comm.Send(my_strip[-1, :].copy(), dest=rank + 1, tag=TAG_DOWN)
```

In MPI Standard 4.1 (Chapter 3, "Point-to-Point Communication", Section "Communication Modes"):
- A standard mode `MPI_Send` is permitted to complete before a matching `MPI_Recv` is posted **only** if the implementation buffers the message (Eager Protocol).
- When message size exceeds the eager threshold, `MPI_Send` behaves as a synchronous rendezvous: it will not return until the matching receive buffer is posted on the destination rank.
- Since every rank executed blocking `Send` operations before executing any `Recv` operations, every process became blocked waiting for its neighbor to post a receive, creating an unresolvable cyclic dependency (Deadlock).

---

## 5. Resolution & Verification

### The Fix
Replace separate blocking `Send` and `Recv` invocations with atomic `comm.Sendrecv`:

```python
# img_mpi_v2_halo.py
if rank > 0:
    comm.Sendrecv(
        sendbuf=strip[0, :].copy(), dest=rank - 1, sendtag=30,
        recvbuf=top_halo, source=rank - 1, recvtag=31)
```

`MPI_Sendrecv` guarantees that the local receive operation is posted before or simultaneously with the send initiation, breaking the circular dependency and preventing deadlocks regardless of message size or network buffer capacity.

### Verification
Testing across the previously failing 8192-pixel workload:

```bash
$ mpirun --hostfile hostfile -n 6 python3 img_mpi_v2_halo.py break_8192 break_8192_out_v2
$ python3 compare.py --serial-prefix break_8192_serial --candidate break_8192_out_v2
np.array_equal: True
Differing pixels: 0 / 2457600
0 differing pixels — exact match with the serial oracle.
```

Three consecutive executions completed deterministically with zero differences against the serial ground truth.
