## Pipeline Parallelism in Inference

In the previous post, [Inference Engine]({% post_url 2025-11-16-inference-engine %}), we built an async inference engine. Adding pipeline parallelism (PP) complicates the control flow, especially around send/recv ordering and shutdown. This post walks through the problem and two ordering designs, then lands on a simple loop structure.

### 1) The Naive Pipeline

Each rank receives tensors, runs the model, and sends the output to the next rank.

```python
class Inferencer:
    def infer(self):
        ctx = self._get_ctx()
        output = self._model.forward(ctx)
        self._send_to_next_rank(output, ctx)
```

The flow is clean but produces crossing dependencies:

![PPDeadlock](/img/PPDeadlock.png)

If send/recv are blocking (e.g., `torch.distributed.send_obj_list` for CPU metadata), the arrows cross and deadlock. GPU tensors can live on separate CUDA streams, but CPU metadata cannot.

### 2) Fixing the Ordering

The fix is to flip the send/recv order on either the first or last PP rank.

#### Option A: First-Rank Special-Case (Preferred)

![GoodPPTimeline](/img/GoodPPTimeline.png)

Only the first rank differs; all other ranks stay naive.

```python
# first rank cycle
def infer(self, batch, should_recv_out):
    ctx = self._prepare_ctx(batch)
    output = self._model.forward(ctx)
    if should_recv_out:
        token_id = self._recv_token_id()
    self._send_ctx(output, ctx)
    if should_recv_out:
        return token_id
```

`should_recv_out` says whether this cycle expects a return token from the last rank. The scheduler already tracks batch movement, so it can provide this flag. Complexity is isolated to the first rank.

#### Option B: Last-Rank Special-Case (Awkward)

![UglyPPTimeline](/img/UglyPPTimeline.png)

The first rank stays naive; the last rank breaks symmetry and becomes stateful.

```python
def infer(self):
    ctx = self._get_ctx()
    if self._last_token_id:
        self._send_token_id(self._last_token_id)
    token_id = self._model.forward(ctx)
    self._last_token_id = token_id
```

Problems:
- Introduces hidden state (`_last_token_id`) that complicates testing.
- Breaks when no new batches arrive; the last token never sends because the loop blocks on `_get_ctx()`.
- Requires extra scheduling logic on non-first ranks, making the code ugly fast.

Because Option A keeps all stateful logic on the first rank (owned by the scheduler), it is the better default.

### 3) Loops on Each Rank

With Option A, the loops look like this:

```python
class Engine:
    def first_rank_loop(self):
        while True:
            next_batch, requests_to_receive = self._scheduler.schedule()
            token_ids = self._inferencer.infer(next_batch, requests_to_receive)
            completed = self._output_processor.process(requests_to_receive, token_ids)
            self._scheduler.cleanup_requests(completed)

class Worker:
    def other_rank_loop(self):
        while True:
            self._inferencer.infer([])
```

All non-first ranks stay simple; the scheduler drives statefulness on the first rank.

### 4) Shutdown Considerations

Simple loop termination via `torch.multiprocessing.Event` is not enough; loops block waiting for `get_ctx()`. Two approaches:

- **Shutdown ctx sentinel:** Send a special ctx through the pipeline to signal exit. This makes `infer` return a shutdown signal, which can be awkward.
- **Queue batches instead of pull:** Pass `next_batch` via `mp.Queue`; each rank calls `prepare_ctx` instead of `get_ctx`. This simplifies `infer` but does extra per-rank prep. It also only works with `torch.multiprocessing`, not multi-node `torch.distributed`.

Given multi-node PP needs `torch.distributed`, the first-rank special-case with scheduler-driven flags remains the pragmatic choice.

### 5) Takeaways

- Deadlock arises when CPU metadata send/recv are blocking with naive ordering.
- Flip ordering on a single rank to break the dependency; prefer doing it on the first rank.
- Keep all scheduling and state on the first rank; keep other ranks stateless and loop-simple.
- Plan shutdown early; avoid designs that force hidden state on non-first ranks.
