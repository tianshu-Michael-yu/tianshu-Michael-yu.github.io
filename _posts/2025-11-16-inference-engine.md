## Inference Engine

This post introduces the core components of a typical inference engine and how they coordinate to keep a GPU saturated while serving requests.

![InferenceEngineStructrue](/img/inference_engine_structure.png)

### Components at a Glance

- **Engine** – entry point that accepts user requests, schedules work, and gathers outputs.
- **Worker** – one process per accelerator (GPU, TPU, etc.); owns the lifetime of a model shard on that device.
- **Inferencer** – GPU-heavy component responsible for preparing device tensors, launching kernels, and sampling tokens.
- **ModelBuilder** – loads the model shard for a worker. Keeping it separate from the inferencer makes unit testing and reloading logic tractable.
- **Scheduler** – decides which requests form the next batch, how tokens are grouped, and how KV-cache blocks are assigned.
- **Block manager** – tracks block usage for paged attention so the scheduler knows which memory regions are free or must be evicted.

Separating these roles keeps the inferencer focused on GPU math. Model creation and sharding are complicated and rarely change once data is on the device, so isolating them behind the model builder simplifies the rest of the system.

### Engine Loop

```python3
class Engine:
    def add_requests(self, requests: List[Request]):
        ...

    def _step(self):
        batch = self._scheduler.schedule()
        self._workers.execute("queue_batch", batch)
    
    def generate(self):
        while True:
            self._step()

    def _update_output(self):
        ...
```

- **`add_requests`** queues new work without blocking the main loop, which lets user traffic arrive asynchronously.
- **`_step`** asks the scheduler for the next batch and broadcasts it to workers through RPC (e.g., `queue_batch`).
- **`_update_output`** is called remotely by workers when they have tokens ready so that the engine can stream results back to clients.

### Worker Responsibilities

```python3
class Worker:
    def queue_batch(self, batch):
        ...
    
    def worker_loop(self):
        while True:
            batch = self._next_batch()
            output = inferencer.infer(batch)
            self._output_processor.process(output)
```

- Workers ingest batches via `queue_batch`, typically invoked by the engine.
- The worker loop is intentionally tight: fetch the next batch, run inference, ship outputs to the output processor.
- Output processors live on the CPU side. They wait for GPU→CPU transfers to finish, perform light post-processing, and RPC results back to the engine or a streaming endpoint.

### Inferencer Responsibilities

```python3
class Inferencer:
    def infer(self, batch):
        ctx = self._prepare_context(batch)
        output = self._forward(ctx)
        return self._sampler.sample(output)
```

- **`_prepare_context`** packs inputs, positions, and attention masks into device tensors and allocates the necessary buffers.
- **`_forward`** launches the actual compute graph on the GPU.
- **`_sampler`** optionally turns logits into the next token using sampling, greedy decoding, or beam search.

## Sync to Async

The most naive workflow keeps every step strictly sequential.

![NaiveTimeline](/img/naive_inference_timeline.png)

- **prep_ctx** – build the context for the forward pass and allocate tensors on the GPU.
- **forward** – CPU launches kernels while the GPU performs the compute.
- **proc_out** – wait for the GPU result, transfer it back, and post-process it on the CPU.

Even though GPUs allow asynchronous kernel launches, the above flow blocks on CPU work before scheduling the next batch because it needs the previous batch’s CPU result. That pause—the “bubble”—has the GPU waiting for more work. Serving a model like qwen3_30b on a single H100 can lose ~20% throughput and per-token latency to this bubble.

### Introducing Asynchronous Scheduling

![AsyncTimeline](/img/async_inference_timeline.png)

- `proc_out` for cycle *i* now processes the output generated in cycle *i − 1*.
- CPU post-processing and scheduling happen while the GPU is already working on the next batch, hiding the bubble entirely when things line up.

### Output Processor Rework

```
class OutputProcessor:
    def __init__(self):
        self._prev_output = None

    def process(self, output):
        if self._prev_output:
            self._prev_output.synchronize()
            self._process(self._prev_output)
        self._prev_output = output
```

Instead of processing the current output immediately, we flush the previous one and stash the current result for the next iteration. Synchronization now happens one cycle later, which keeps the GPU busy.

### Additional Adjustments

- **Scheduler/context prep** – when building the context for cycle *i*, the result from *i − 1* may not exist yet, so we allocate space in advance and patch it later.
- **Termination logic** – because cycle *i* no longer examines its own output, it doesn’t know whether it emitted an end token; we detect completion one cycle later and naturally run one extra iteration.
- **Back-pressure** – asynchronous pipelines need guardrails (queue depths, outstanding batch caps) so that workers don’t race too far ahead of what the engine or clients can consume.
