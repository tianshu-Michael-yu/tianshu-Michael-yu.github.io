# LFM2-VL Performance: Two Hidden Host↔Device Syncs in Metadata Building

When profiling `lfm2-vl`, most of the “why is my GPU idle?” time showed up *outside* the model kernels—in the attention **metadata builder**. This post documents two concrete bottlenecks I hit, how to spot them in a trace, and the fixes that restored async scheduling.

The hot path looked like this (simplified):

```python
class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ShortConvAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        has_initial_states_p = None
        if num_prefills > 0:
            has_initial_states_cpu = (
                common_attn_metadata.num_computed_tokens_cpu[
                    num_reqs - num_prefills : num_reqs
                ]
                > 0
            )
            has_initial_states_p = has_initial_states_cpu.to(query_start_loc.device)  # Problem 1

            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc_p)  # Problem 2
            )

        elif (
            num_decodes > 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:num_decode_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

        return ShortConvAttentionMetadata(
            query_start_loc=query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_p=has_initial_states_p,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
```

Both issues come from the same general rule:

> If you want **full async scheduling**, be paranoid about **implicit CPU↔GPU sync** and about using **pageable host memory** for copies.

---

## Background: what is `has_initial_states_p` actually checking?

During serving, a “sequence at iteration \(N\)” can be viewed like:

```
|---------- N-1 iteration --------|
|---------------- N iteration ---------------------|
|- tokenA -|......................|-- newTokens ---|
|---------- context_len ----------|
|-------------------- seq_len ---------------------|
                                  |-- query_len ---|
```

`has_initial_states_p` is trying to determine whether we already have an *initial state* for short-conv attention, which is essentially checking:

- **context_len > 0**  (i.e., this request is not “pure prompt from scratch”)

In other words, it wants a per-request boolean derived from `query_start_loc` and `seq_lens`.

---

## Problem 1: “tiny” CPU→GPU copy that still breaks async

Here is what I originally had:

```python
if num_prefills > 0:
    has_initial_states_cpu = (
        common_attn_metadata.num_computed_tokens_cpu[
            num_reqs - num_prefills : num_reqs
        ]
        > 0
    )
    has_initial_states_p = has_initial_states_cpu.to(query_start_loc.device)
```

On the trace, this showed up as a noticeable sync/copy point:

![has_initial_states_cpu_to](/img/has_inital_states_cpu_to.png)

### Fix A: if you must H→D, use pinned memory + non-blocking

If the value truly must originate on CPU, at minimum make the transfer async-friendly:

```python
has_initial_states_cpu = (...).pin_memory()
has_initial_states_p = has_initial_states_cpu.to(
    query_start_loc.device, non_blocking=True
)
```

This dropped `short_conv_attn: build: has_initial_states_cpu.to` down to **~28.367 μs** in my run.

### Fix B (better): don’t go to CPU in the first place

The bigger “aha” was that `num_computed_tokens_cpu` is itself suspicious. In fact, in the codebase it is explicitly deprecated because it risks implicit sync:

```python
@property
@deprecated(
    """
Prefer using device seq_lens directly to avoid implicit H<>D sync which breaks full
async scheduling. If a CPU copy is needed, it can be derived from
query_start_loc_cpu and seq_lens.
Will be removed in a future release (v0.14.0)
"""
)
def num_computed_tokens_cpu(self) -> torch.Tensor:
    if self._num_computed_tokens_cpu is None:
        query_seq_lens = self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
        self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
    return self._num_computed_tokens_cpu
```

But we don’t need any CPU tensor here. `query_start_loc` and `seq_lens` already exist on device, so compute the boolean on GPU:

```python
query_seq_lens = (
    common_attn_metadata.query_start_loc[1:]
    - common_attn_metadata.query_start_loc[:-1]
)
context_lens = common_attn_metadata.seq_lens - query_seq_lens

has_initial_states_p = context_lens[num_reqs - num_prefills : num_reqs] > 0
```

This completely removes the host dependency for Problem 1.

---

## Problem 2: a 30 ms `cudaMemcpyAsync` that shouldn’t exist

The next spike was much worse: a single `cudaMemcpyAsync` taking **~30.796 ms**.

![compute_causal_conv1d_metadata](/img/compute_causal_conv1d_metadata.png)

When a CUDA API call is in **milliseconds** (not microseconds), it usually means you accidentally introduced a synchronization point or forced a slow path.

Zooming in, the GPU-side copy was:

- `Memcpy DtoH (Pageable)` on the **main compute stream**

![compute_causal_conv1d_metadata_relationship](/img/compute_causal_conv1d_metadata_relationship.png)

Two red flags:

- **DtoH on the main stream**: it blocks the CPU from enqueuing later kernels until the copy completes (because the runtime must ensure the source is ready).
- **Pageable host memory**: true async DMA requires pinned memory. With pageable memory, CUDA often has to:
  - allocate/reuse an internal pinned staging buffer
  - synchronize to ensure the device source is ready
  - DMA into pinned memory
  - memcpy pinned → pageable on CPU

### Root cause

The culprit was inside `compute_causal_conv1d_metadata`:

```python
def compute_causal_conv1d_metadata(query_start_loc_p: torch.Tensor):
    # Needed for causal_conv1d
    seqlens = query_start_loc_p.diff().to("cpu")  # DtoH copy to pageable memory
    nums_dict = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    device = query_start_loc_p.device
    for BLOCK_M in [8]:
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist)                 # HtoD from pageable
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist)  # HtoD from pageable
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return nums_dict, batch_ptr, token_chunk_offset_ptr
```

There are *three* expensive transfers here:

- **DtoH**: `query_start_loc_p.diff().to("cpu")`
- **HtoD**: `batch_ptr.copy_(mlist)`
- **HtoD**: `token_chunk_offset_ptr.copy_(offsetlist)`

### Fix: keep the input CPU-side, and make HtoD transfers pinned + non-blocking

First principle: ask if the movement is necessary.

In this case, `compute_causal_conv1d_metadata` is building **CPU-side lists** anyway; it can just take CPU input and avoid the DtoH entirely:

```python
def compute_causal_conv1d_metadata(query_start_loc_cpu: torch.Tensor, device: torch.device):
    ...
```

Then for the remaining HtoD copies:

- allocate the CPU tensors in **pinned memory** (`pin_memory()`)
- use **non-blocking** copies (`copy_(..., non_blocking=True)`)

After these changes, the trace looked like this:

![fixed_compute_causal_conv1d_metadata.png](/img/fixed_compute_causal_conv1d_metadata.png)

And `compute_causal_conv1d_metadata` dropped to **~292.806 μs**, with the async memcpy (red arrow) becoming negligible.

---

## Takeaways

- **Prefer device-side derivations** when you already have device tensors (`seq_lens`, `query_start_loc`). Avoid “convenience” CPU properties that hide sync.
- If you must transfer CPU↔GPU in the hot path:
  - use **pinned memory** (`pin_memory()`)
  - use **non-blocking** transfers (`to(..., non_blocking=True)` / `copy_(..., non_blocking=True)`)
- A `cudaMemcpyAsync` taking **milliseconds** is often a sign you triggered a sync or a pageable-memory slow path. Always check whether it’s `DtoH (Pageable)` / `HtoD (Pageable)` and on which stream it runs.
