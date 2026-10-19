# AI Interaction & Verification Log

**Author:** Moaaz Sherif  
**Methodology:** AI Proposes, Engineer Verifies

---

## Round 1: Deadlock Diagnosis

### Prompt Sent to Assistant
```text
I'm running a 3-container Docker Compose cluster (node0, node1, node2) and
launch: mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py
from node0, with mpi4py on Open MPI 4.1.4. All containers are Up (healthy).

Expected: the job finishes and writes an output file.
Actual: The job hangs after printing before Send markers, and after 20s timeout kills it with exit code 124. No error message.
Small image (width 256): it finishes cleanly and fast. Large image (width 8192): it hangs indefinitely.

Evidence:
$ docker compose ps
node0   scx61-week2-node0   Up (healthy)
node1   scx61-week2-node1   Up (healthy)
node2   scx61-week2-node2   Up (healthy)

$ mpirun --hostfile hostfile -n 6 hostname
node0
node0
node1
node1
node2
node2

$ bash break_deadlock.sh 20
[rank 1 on node0] before Send
[rank 2 on node1] before Send
[rank 3 on node1] before Send
[rank 4 on node2] before Send
[rank 0 on node0] before Send
[rank 5 on node2] before Send
=== TIMEOUT after 20s - mpirun was killed. ===

Relevant code:
    print(f"[rank {rank} on {hostname}] before Send", flush=True)

    if rank > 0:
        comm.Send(my_strip[0, :].copy(), dest=rank - 1, tag=TAG_UP)
    if rank < size - 1:
        comm.Send(my_strip[-1, :].copy(), dest=rank + 1, tag=TAG_DOWN)

    print(f"[rank {rank} on {hostname}] after Send", flush=True)

    top_halo = np.empty(width_for_halo, dtype=np.uint8)
    bottom_halo = np.empty(width_for_halo, dtype=np.uint8)

    if rank > 0:
        comm.Recv(top_halo, source=rank - 1, tag=TAG_DOWN)
    else:
        top_halo = my_strip[0, :].copy()

    if rank < size - 1:
        comm.Recv(bottom_halo, source=rank + 1, tag=TAG_UP)
    else:
        bottom_halo = my_strip[-1, :].copy()

Give me: 1) your top 2-3 hypotheses, ranked; 2) one command per hypothesis to
confirm or reject it; 3) the exact output that would confirm vs reject each;
4) the smallest fix (not a rewrite); 5) one command that proves the fix worked.
Do not say "check your MPI setup".
```

### Response Received
The assistant correctly identified that the 8192-byte messages exceed the default eager buffer limit, causing `comm.Send` to block and deadlocking the processes in a circular wait. It proposed verifying this by substituting `comm.Send` with `comm.Ssend` on a small 256-wide image.

### Experimental Verification
Running the assistant's proposed `Ssend` test confirmed the hypothesis:
```bash
$ timeout 20 mpirun --hostfile hostfile -n 6 python3 v3_ssend_test.py small256 small256_out
rc=124
```
The test reproduced the hang on small inputs, proving that eager buffering had previously masked the bug.

---

## Round 2: Critical Evaluation of Claims

We evaluated three potential claims:

| Claim | Proposal | Experimental Test | Result & Verdict |
|---|---|---|---|
| **Claim A** | Docker network/DNS is flaky; restart Docker daemon. | `docker compose exec node0 ping -c 3 node1` | `0% packet loss`. Network is healthy. **REFUTED**. |
| **Claim B** | `sshd` is unreachable on worker nodes. | `mpirun --hostfile hostfile -n 6 hostname` | Printed 6 lines across all 3 nodes cleanly. **REFUTED**. |
| **Claim C** | Increase MPI eager buffer (`--mca btl_vader_eager_limit 16384`). | Tested on 8192-width (passed, `rc=0`), then scaled to 20,000-width (`rc=124`). | Fails under further scaling. **HIDES THE SYMPTOM**. |

### Conclusion
Increasing buffer limits only shifts the failure boundary higher without eliminating the circular wait. The only architecturally sound solution is implementing `comm.Sendrecv`.
