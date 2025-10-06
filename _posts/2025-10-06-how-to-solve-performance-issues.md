## How to solve performance issues

Let's start with a profile of the inference engine. 

![Alt text](/img/dp_2_tp_4_nsys_timeline_1.png)

This figure shows the nsys profiling result for the rollout phase of one RL step.
We’re running on a machine with 8× H100 GPUs, hosting two inference engines.
Each engine uses tensor parallel size 4 (TP = 4).

---

### Identifying the Symptom

To understand why, we need to zoom in on one inference engine.
However, nsys doesn’t label which processes belong to which engine.
Luckily, model parallelism gives us clues: in tensor parallelism, each shard performs two all-reduces per decode layer, so their forwards should finish around the same time. Let's pin those processses.

![Alt text](/img/dp_2_tp_4_nsys_timeline_2.png)

---
### Drilling Down

Take TP rank 3 as an example.

![Alt text](/img/dp_2_tp_4_nsys_timeline_3.png)

We can see the CUDA Graph launch is delayed — that’s the direct cause of the bubble.
Inspecting the CPU side more closely (by expanding the Threads view in nsys), we find that the CUDA Graph delay stems from gen_context preparation, which itself is blocked by a long zmq_bcast_ctx_meta call.

---
### Forming Hypotheses

Two natural explanations come to mind:
	1.	Interprocess communication (IPC) is too slow.
	2.	Python object serialization takes too long.

Between the two, (2) seems more likely — IPC here uses shared memory (ipc://), so bandwidth shouldn’t be a bottleneck.
If (2) is true, it means the Python objects inside gen_ctx_meta are simply too large.
---

### Verifying the Cause
Inspecting gen_ctx_meta confirms this: it includes slots_mappings, whose size scales with batch_size × seq_len.
If serialization takes ~3 ms, deserialization costs another 3 ms — that’s 6 ms of stall per step.
With long sequences, the delay compounds, causing non-scheduler ranks to launch CUDA Graphs late and leaving visible idle bubbles.
---

### Fix and Result

![Alt text](/img/dp_2_tp_4_nsys_timeline_4.png)

As you can see, after we remove `slots_mappings` from the data to be sent, the cudagraph for all tp rank starts at approximately the same time and there're no bubbles on the timeline.
---

### Takeaway

In performance debugging, look for serialization and coordination overheads hiding between GPU launches.
A few milliseconds of CPU-side latency per process can easily cascade into multi-millisecond GPU stalls across the cluster.
