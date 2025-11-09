# Sequence Parallelism

In our previous post, [Model Parallelism]({% post_url 2025-10-28-model-parallelism %}), we focused on sharding model weights to make massive networks fit on limited memory. That strategy helps online serving, where batches are tiny and weights dominate the footprint. During training, however, each batch often contains far more token activations than parameters, so splitting models alone is not enough.

A batch can contain multiple sequences, and a sequence is an ordered list of tokens whose relationships we care about. When we parallelize training we have two axes to pick from: shard across independent sequences (traditional data parallelism) or shard within a sequence (sequence parallelism). This post concentrates on the latter and explains how different algorithms handle the dependency structure inside a single long sequence.

## Attention

Most transformer components treat tokens identically, but multi-head attention must preserve where each token sits in the context window. Positional embeddings inject order, and the scaled dot-product step consumes it. That is why sequence parallelism discussions usually start with attention.

![Attention](/img/attention_compute_graph.png)

The compute graph above shows attention for a single sequence (batch size 1). Squares are tensors with shapes annotated in brackets, and circles are operations. Sequence length `N` denotes the number of tokens, and `head count` is the number of attention heads. Because attention mixes information along the sequence dimension, we cannot simply split the tokens without introducing communication.

## Ulysses

![Ulysses](/img/ulysses_compute_graph.png)

Ulysses shards the input along the sequence dimension into `P` pieces so that each GPU stores only `N/P` tokens. If every GPU ran attention independently, it would only capture dependencies inside its own shard. To avoid that, Ulysses uses an all-to-all exchange that converts the shard axis from sequence positions into attention heads.

![UlyssesAll2All](/img/ulysses_all_to_all.png)

In the example, tokens `a`, `b`, `c`, and `d` are split across four GPUs. Before communication, GPU0 holds all projections of token `a`, GPU1 holds token `b`, and so on. The all-to-all redistributes the projections so that each GPU ends up with the same head across every token (GPU0 receives `b0`, `c0`, `d0`, etc.). The per-GPU communication volume is roughly $2\frac{h_{size}}{P^2}(P-1) \approx 2\frac{h_{size}}{P}$, so increasing the number of sequence-parallel shards reduces the cost per rank.

After this reshuffle, each GPU can run full attention over the entire sequence for a subset of heads to obtain $P_h$. A second all-to-all converts the results back so that every rank has the portion of $P$ that corresponds to its original sequence shard.

Ulysses is attractive because the algorithm and implementation are simple, and the all-to-all volume is modest. The downsides are (1) the degree of sequence parallelism cannot exceed the number of attention heads, and (2) all-to-all is latency sensitive and expects a supportive network topology.

## RingAttention

Ring-attention is essentially the multi-GPU version of FlashAttention. The insight is that we can compute partial attention results locally and then renormalize them to recover the global answer.

![SplitKV](/img/split-kv.png)

Suppose $k_1, k_2$ live on GPU1 and $k_3, k_4$ live on GPU2. On GPU1 we compute

$$
\begin{aligned}
p_1 &= \text{softmax}([q k_1^\top, q k_2^\top]) [v_1; v_2] \\
&= \frac{1}{z_1}[\exp(q k_1^\top), \exp(q k_2^\top)] [v_1; v_2] \\
z_1 &= \exp(q k_1^\top) + \exp(q k_2^\top)
\end{aligned}
$$

and similarly on GPU2 to obtain $p_2$. The goal is the global result

$$
\begin{aligned}
p &= \frac{1}{z}[\exp(q k_1^\top), \exp(q k_2^\top), \exp(q k_3^\top), \exp(q k_4^\top)] [v_1; v_2; v_3; v_4] \\
z &= \exp(q k_1^\top) + \exp(q k_2^\top) + \exp(q k_3^\top) + \exp(q k_4^\top)
\end{aligned}
$$

We can rewrite it as $p = \frac{z_1}{z} p_1 + \frac{z_2}{z} p_2$, meaning local attention followed by a renormalized sum recovers the full answer. The catch is that, in practice, $q$ is also sharded, so we need a systematic way to expose every query chunk to every key/value chunk.

![RingAttentionRotation](/img/ring-attention-rotation.png)

$P_{ij}$ means the partial result calculated from ith query chunk and jth kv chunk. The $\text{sum}^*$ arrow in the graph is renormalized summation $\frac{z_1}{z}p_1 + \frac{z_2}{z}p_2$ that we showed above.

In RingAttention each GPU keeps its query chunk but passes key-value (KV) chunks around the devices in a ring. During iteration 1, GPU1 has Q1 with KV1, GPU2 has Q2 with KV2, etc., producing partial results $P_{11}, P_{22}, \ldots$. On iteration 2 we rotate the KV chunks (KV1 moves to GPU2, KV2 to GPU3, …), enabling the computation of $P_{21}, P_{32}, P_{43}, P_{14}$. After `P` iterations every query chunk has interacted with every KV chunk, and a final rotation restores the KV shards to their original owners.

![KV-rotate](/img/KV-rotate.gif)

Each KV transfer costs $4 \frac{h_{size}}{P}$ per GPU, and we perform `P` rotations, so the total per-rank volume is $4 h_{size}$—higher than Ulysses. The advantage is that KV transfers overlap nicely with compute; while GPU1 sends KV1 to GPU2 it can simultaneously process the next partial result. With good overlap the communication becomes almost free, RingAttention scales to any number of shards (it is not limited by head count), and it only needs point-to-point links with modest topology requirements. The major drawback is algorithmic complexity.

## Context Parallelism

Context parallelism (CP) applies the RingAttention idea to inference workloads such as long-context summarization or long-chain reasoning. In inference we maintain a KV cache and step through decoding token by token. Batches are tiny for latency reasons, so queries are small while the KV cache grows with sequence length. Rotating KV chunks while keeping queries fixed would therefore be wasteful.

The "Context Parallelism for Scalable Million-Token Inference" paper proposes rotating the queries instead (Ring Pass-Q) and keeping the KV cache stationary. When we do need to move the cache, we use Ring Pass-KV. Together these strategies let inference systems stretch to million-token contexts without giving up the benefits of sequence parallelism.
