# Model Parallelism
Model parallelism shards a single model across multiple devices so we can train or serve networks that exceed the memory or compute budget of one GPU. This post organizes the three common flavors -- tensor, pipeline, and expert parallelism -- and then looks at how they combine in real-world inference stacks.

## Core Parallel Strategies

### Tensor Parallelism
Consider a single linear layer $Y = X A$ with weight matrix $A = [A_1, A_2]$. Splitting by columns lets us compute the outputs independently:

$$
[Y_1, Y_2] = [X A_1, X A_2]
$$

Each shard $A_i$ sits on a different GPU, so the multiplications happen in parallel. Tensor parallelism (TP) reduces latency for wide layers because identical operations run concurrently on separate devices. Practically, TP pairs well with high-bandwidth links such as NVLink; across slower interconnects the cost of synchronizing partial results can dominate.

### Pipeline Parallelism
Pipeline parallelism (PP) slices the model by depth instead of width. If a network is composed of stages $A$ and $B$, the forward pass applies them sequentially:

$$
Y = X A,\qquad Z = Y B
$$

With PP, GPU0 runs stage $A$ and streams the intermediate activations to GPU1, which runs stage $B$. After the pipeline is filled, multiple inputs flow simultaneously through different stages:

| Time | 0       | 1       | 2       | 3       |
|------|:-------:|:-------:|:-------:|:-------:|
| GPU0 | $X_1A$  | $X_2A$  | $X_3A$  |         |
| GPU1 |         | $Y_1B$  | $Y_2B$  | $Y_3B$  |

PP typically does not shrink per-sample latency -- operations still happen in series, now with added communication cost. Its strength is throughput: keeping all devices busy when large batches stream through the model.

### Expert Parallelism
![Mixture-of-Experts routing](/img/moe_layer.png)

Expert parallelism (EP) targets Mixture-of-Experts (MoE) architectures. An MoE layer owns many experts $A_1, A_2, \dotsc, A_n$, but a router activates only a subset per token. In EP we place different experts on different GPUs (or groups of GPUs). Routing sends the token to the device that hosts the selected expert, performs the computation, and gathers the results.

Load balancing is the central challenge: if tokens concentrate on a single expert, one GPU becomes a bottleneck while others idle. When the routers distribute tokens evenly, EP can reduce latency for large MoE layers because multiple experts run concurrently. When they do not, the extra all-to-all communication and synchronization can hurt performance.

## Communication Patterns in Transformer Blocks
Real transformer stacks combine tensor and pipeline parallelism. The schematic below highlights how a transformer block is partitioned for TP, while PP slices the network across blocks.

![Tensor-parallel transformer block](/img/tensor_parallelism.png)

In TP, each transformer block issues two all-reduce operations -- one inside self-attention, another inside the MLP. Assuming TP size $n$ and hidden dimension $h_{\text{size}}$, the per-GPU cost of a ring all-reduce is roughly $4 (n-1) \frac{h_{\text{size}}}{n}$. Because every block triggers this pattern, total communication scales with the number of layers $k$: about $8 k h_{\text{size}}$ per GPU.

PP communication is simpler. Each boundary between pipeline stages introduces a send/receive pair whose size is $\approx h_{\text{size}}$. The number of such transfers depends on the number of pipeline stages $m$ rather than the number of layers. Since TP's all-reduces are heavier and more frequent, we usually keep TP within a node where NVLink mitigates the cost, and rely on PP across nodes over slower RDMA links.

![Intra- and inter-layer parallelism](/img/intra-and_inter-layer_parallelism.png)

## Latency in Prefill vs. Decode
![Pipeline vs. continuous pipeline](/img/PPvsCPP.png)

Pipeline parallelism shines during the prefill phase of large language model serving. When the full prompt is known, we can keep the pipeline saturated by sending tokens back-to-back, which improves time-to-first-token (TTFT). During autoregressive decoding, however, each new token depends on the previous output; PP cannot hide the serialized dependency, so it offers little per-token latency gain.

![Multi-node expert parallelism](/img/multi_node_ep.webp)

Expert parallelism introduces another layer of synchronization. EP relies on all-to-all communication to route tokens from each data-parallel (DP) rank to the right experts and back again. With $l$ EP ranks and a router that selects `top_k` experts per token, the per-GPU volume is about $\frac{top_k \, h_{\text{size}}}{l}$. Theoretically, larger expert groups reduce the volume, but in practice all-to-all is difficult to optimize. A single slow rank delays the entire group, so uneven routing can cripple throughput.

## Practical Considerations
Many modern models, such as Qwen3, use Grouped Query Attention (GQA) where a small set of key/value heads is shared across many query heads. If TP shards the attention heads beyond the number of key/value groups, each GPU must replicate the key/value weights and cache, wasting memory and synchronization time. For this reason practitioners often cap TP at the number of KV heads; with Qwen3-235B-A22B that means TP <= 4.

Serving budgets underscore these trade-offs. Each NVIDIA H100 provides 80 GB of memory. With 235 B parameters stored in bfloat16, the weights alone consume $235 \times 2 \text{ bytes} \approx 470 \text{ GB}$. If deployed on 1 node of 8 H100, each GPU only have about 20GB for kv cache, which is not enough for efficient serving.  Deploying across two nodes with eight H100s each gives 16 GPUs total is a more reasonable choice. Choosing TP = 4 and PP = 4 creates 16 shards -- one per GPU -- which fits the weights while leaving room for key/value cache.

In practical online serving, prefill and decode workload is often disaggregated. Prefill-heavy nodes lean on PP for throughput, while decode-focused nodes stay conservative with TP to avoid redundant KV replication.

## Takeaways
- Use TP to split wide layers, but keep it within nodes to leverage high-bandwidth links.
- Combine PP with large batch sizes or prefill workloads to maximize utilization and lower TTFT.
- Deploy EP only when MoE routing is well balanced; otherwise the all-to-all overhead can dominate.
- Tune TP and PP jointly based on memory, interconnect, and architectural constraints such as GQA.