# LFM2 VL Performance Issues

Currently, most of the perfomance issue for `lfm2-vl` model concentrates in this class. 

``` python
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
            has_initial_states_p = has_initial_states_cpu.to(query_start_loc.device) # Problem 1

            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc_p)
            ) # Problem 2

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

        attn_metadata = ShortConvAttentionMetadata(
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
        return attn_metadata
```



## Problems 1

![has_inital_states_cpu_to](/img/has_inital_states_cpu_to.png)

``` python
    class ShortConvAttentionMetadataBuilder:
        def build(
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> ShortConvAttentionMetadata:
            ...
            if num_prefills > 0:
                has_initial_states_cpu = (
                    common_attn_metadata.num_computed_tokens_cpu[
                        num_reqs - num_prefills : num_reqs
                    ]
                    > 0
                ) # should be pinned
                has_initial_states_p = has_initial_states_cpu.to(query_start_loc.device) # should turn on non_blocking copy
            ...
```

The correct way
```
    has_inital_states_cpu = (...).pin_memory()
    has_inital_states_p = has_initial_states_cpu.to(query_start_loc.device, non_blocking=True)
```

After this revision the `short_conv_attn: build: has_initial_states_cpu.to` is reduced only taking 28.367 μs. 

But taking a closer look

```
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
            query_seq_lens = (
                self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
            )
            self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
        return self._num_computed_tokens_cpu
```

Usually a sequence can be partitioned into the following view.


```
|---------- N-1 iteration --------|
|---------------- N iteration ---------------------|
|- tokenA -|......................|-- newTokens ---|
|---------- context_len ----------|
|-------------------- seq_len ---------------------|
                                  |-- query_len ---|

```

`has_inital_states_p` wants to determine whether we already has a inital state which is basically whether context_len > 0. Since `query_start_loc`
and `seq_lens` are both device tensor, we don't ever need to move them. 

```
    query_seq_lens = (
        common_attn_metadata.query_start_loc[1:] - common_attn_metadata.query_start_loc[:-1]
    )
    # Compute context_lens on GPU
    context_lens = common_attn_metadata.seq_lens - query_seq_lens
    # Slice the prefill portion and check if > 0

    has_initial_states_p = (
        context_lens[num_reqs - num_prefills : num_reqs] > 0
    )
```

## Problem 2

![compute_causal_conv1d_metadata](/img/compute_causal_conv1d_metadata.png)

This gigantic `cudaMemcpyAsync` takes 30.796 ms. When an cudaAPI call takes on the order of ms in stead of μs, there's something wrong.

![compute_causal_conv1d_metadata_relationship](/img/compute_causal_conv1d_metadata_relationship.png)

If you zoom in and look at the corrresponding memory operation on GPU, you see its `Memcpy DtoH (Pageable)`. There're two things that are immediately alarming. 

First, it's an `DtoH` copy on the main compute stream. It will block cpu from dispatching later kernels until the current Memcpy operation is done.

Second, it's `Pageable`. Because pageable host memory cannot be the target of true async DMA. CUDA has to do something like:
	1.	allocate / reuse an internal pinned staging buffer
	2.	ensure the source data is ready (often requires sync with prior GPU work on that stream)
	3.	do the DMA into pinned memory
	4.	copy from pinned → pageable (CPU memcpy), then return
These are expensive operations. 

Looking at the source code.

``` python
def compute_causal_conv1d_metadata(query_start_loc_p: torch.Tensor):
    # Needed for causal_conv1d
    seqlens = query_start_loc_p.diff().to("cpu") # !DtoH copy without pinned memory
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
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(  
                    MAX_NUM_PROGRAMS
                ).fill_(PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist) # !HtoD copy without pinned memory
        token_chunk_offset_ptr[  
            0:mlist_len
        ].copy_(offsetlist) # !HtoD copy without pinned memory
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr 

    return nums_dict, batch_ptr, token_chunk_offset_ptr
```

The first thing is always ask whether movement of data is necessary.

Actually the input could be a cpu tensor.

``` python
def compute_causal_conv1d_metadata(query_start_loc_cpu: torch.Tensor, device: torch.device)
```

This solved the DtoH copy.

For those HtoD copy, we can just do `cpu_tensor.pin_memory()` and change `copy_(cpu_tensor)` to `copy_(cpu_tensor, non_blocking=True)`. These changes ensure
the HtoD memcpy don't block the cpu and we dispatch asynchronously.

![fixed_compute_causal_conv1d_metadata.png](/img/fixed_compute_causal_conv1d_metadata.png)

In this graph, after the fix, the `compute_causal_conv1d_metadata` now takes only 292.806 μs and the async memcpy as indicated by the red arrow now takes a negliable amount of time. 
