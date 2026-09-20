# K3 wide EP with MorI EPv2

The native TP8/DPA path can opt into MorI EPv2 with `ATOM_MORI_V2=1`.
The adapter passes the physical local EP size, selects HIP for multi-node groups,
and preserves the native TP sender-padding and mixed-prefill receive bounds.
The existing default transport remains EPv1 until EPv2 is explicitly enabled.

Use MorI `03a1326af0a2584d18dbc0166e8686d981d1c7ed` for the validation baseline.
Installing a newer MorI alone is insufficient: omitting `gpu_per_node` makes
its op-layer constructor select the intranode path.

| Setting | Default | Meaning |
| --- | --- | --- |
| `ATOM_MORI_V2` | `0` | Opt into the EPv2 adapter |
| `ATOM_MORI_V2_QPS_PER_PE` | `2` | RDMA QPs per peer; part of the cached transport identity |
| `ATOM_MORI_V2_INTERNODE_KERNEL` | `auto` | `auto`, `v2`, or `v2_ll` |
| `ATOM_MORI_V2_LL_MAX_TOKENS` | `512` | Auto-family cutoff in the group-wide maximum sender-row bound |
| `MORI_EP_DISP_GEOM` | upstream | Upstream compile-time dispatch override: blocks,RDMA blocks,warps |
| `MORI_EP_COMB_GEOM` | upstream | Upstream compile-time combine override: blocks,RDMA blocks,warps |

Auto mode compiles both families once. Before each dispatch/combine pair, ATOM
selects the same family on all ranks using CPU DP metadata (and TP slicing).
Using each rank's local row count can select different protocols in a mixed
prefill/decode batch. Graph capture records that shared choice without a GPU
counter read. Missing batch metadata conservatively selects the general family.

K3 uses BF16 dispatch and BF16 combine in this adapter; A8W4 expert compute is
separate. The actual model input dtype is passed to MorI without guessing it
from item size. FP4 transport, quantized combine and fused MegaMoE are not
supported on this internode path. `ATOM_MORI_V2_FUSED=1` and forced FlyDSL are
rejected for multi-node EP. The local MegaMoE implementation remains separate.

ROCm 7.2.4 requires a matching HIP runtime rebuild for CCO multi-node allocation:
CLR commit `be8bcd0595ae02f4add19aea1ffd7db2a336d90d` fixes sub-buffer coverage in
`hipMemSetAccess`. The upstream uncached VMM pool routing fix also matters for
latency. Validation uses a task-local `libamdhip64.so.7` via `LD_PRELOAD`
and puts that runtime directory first in `LD_LIBRARY_PATH`, with an unversioned
`libamdhip64.so` symlink. The latter also resolves ATOM's ctypes stream helper;
preloading alone can otherwise load a second, stock HIP image. No host or
production library is replaced. All test hosts have both
`CONFIG_DMABUF_MOVE_NOTIFY=y` and `CONFIG_PCI_P2PDMA=y`.

The unmodified collapsed-CQ build stalled during repeated EP32 padded-batch
replay. An isolated normal-CQ-ring compatibility patch covers both CCO's separate
poller and the shmem poller, preserves the upstream counter-wrap fixes, and is
enabled with `MORI_BNXT_CQ_RING=1`. The tested package, runtime and JIT cache are
isolated from the upstream package. The exact collapsed-CQ failure mechanism
remains unproven; the ring results establish a tested workaround.

A collapsed CQ overwrites a single completion slot with cumulative progress.
A normal ring retains successive completion entries; the consumer checks phase
ownership, advances its index and acknowledges consumed entries to the NIC.
Recent upstream serial-counter fixes improve collapsed polling without turning
it into a ring. Creating a ring alone is insufficient: each active poller must
consume it correctly.

Validation on 16 and 32 MI355X GPUs passed exact expert-dependent dispatch and
combine checks, mixed-DPA adapter cases (including the family cutoff), repeated
graph replay, and all 48 primary QP/family jobs. With capacity 128 and eight rows
per rank, QP1 reduced low-latency dispatch+combine median time from 98.06 to
91.46 microseconds on EP16, and from 152.50 to 142.22 microseconds on EP32.
These timings exclude expert compute. Capacity 8192 has materially higher
communication cost, so small-buffer results cannot predict serving throughput.

Actual K3 serving passed basic answer probes and native Compass scoring for
EP16/EP32 at QP1/QP2. Native excluded warmup lasted 90 seconds (five-second load
ramp), followed seamlessly by 60 seconds of profiling. Observed output TPM was
62,421 / 53,607 for EP16 QP2 / QP1 and 74,785 / 68,227 for EP32 QP2 / QP1.
All four points failed the predeclared stability check. All five trace request
shapes were issued, but only chat and tool-chat requests completed before cutoff.
These are short transient observations, not steady-state capacity or proof of a
whole-model QP gain. The default remains QP2. Longer matched, stable runs are
needed before a global tuning recommendation or an EPv2-over-EPv1 speedup claim.
