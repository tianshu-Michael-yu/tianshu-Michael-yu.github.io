# Sequence Parallelism

In our previous post, [Model Parallelism]({% post_url 2025-10-28-model-parallelism %}), we introduced model parallelism. It shards the size of the model. Since in online serving, the number of tokens in a batch is small, model weight dominates the memory consumption. Model parallelism is more common in serving. In training, however, the amount of data in a batch can be much larger than the weight of the model, so we need to shard the data. There're two dimensions when we parallelize the data. A sequence is a sequence of tokens, which we interested in their relationship. In a batch we can have mulitple sequences. We can parallelize across different sequence, which is data parallel and parallelization within sequence, which is sequence parallelism. We will discuss sequence parallelism in this post.

## Attention

All other component in the transformers treats each token the same way except attention. Specifically the positional embedding and scaled dot product are the components that inject positional information and utilize those information.

![Attention](/img/attention_compute_graph.png)

This is a compute graph of attention. The squares are tensors and the circles are operations. The square brackets on each squares describes the shape of the tensor. The sequence length being N means that the sequence have N tokens. Head count refers to the number of attention head. Here we only have one sequence, i.e. batch size is 1. 

## Ulysses

![Ulysses](/img/ulysses_compute_graph.png)

In ulysses, our input is sharded to P different pieces along the sequence dimension. Each GPU only have N/P number of tokens. If we continue doing attention on each GPU without communication, we will only be doing attention for part of the sequence, we will lose the information about the relationship between tokens in different shards. Ulysses change the shard along the sequence dimension to shard along the head dimension through all-to-all communication operator.

Here is a picture of what ulysses all-to-all does. 

![UlyssesAll2All](/img/ulysses_all_to_all.png)

Here the inputs are the projections of the four tokens: a, b, c, d. For a, a0 is the projection of the 0th q head, a2 is the projection of 1th q head, etc. All the projections of a is on GPU 0, all the projections of b is on GPU 1. After the all-to-all, GPU0 gets b0, c0, d0 from GPU1, 2, 3. Similarly for other GPU. Instead of having all the projections of the same token, each GPU now haves all the tokens projected through the same q head. So the communication cost per GPU is $2\frac{h_{size}}{P^2}(P-1) \approx 2\frac{h_{size}}{P}$. So the benefit of using ulysses is that as you increase the number of sequence parallel rank, the communication cost per GPU is going to decrease.

After all-to-all, on each GPU, ulysses can do full-attention for the entire sequence but only with some of the heads to get $P_h$. In order to get the result $P$ of full-attention for the part of the sequence on each rank, the ulysses does the second all-to-all.

The advantage of ulysses lies in its implicity. As you will see, the algorithm for ring-attention is much more complex. Also, the communication overhead is relatively low. The disadvantages of ulysses are two fold. First, the degree of sequence parallel is limited by the amount of attention heads. Second, all-to-all communication is sensitive to latency and has certain requirements for network topology. 

## RingAttention

The ring-attention is actually a multi-gpu version of flash-attention. I will show some intuition behind the algorithm. 

![SplitKV](/img/split-kv.png)

k1, k2 is on GPU 1 and k3, k4 is on GPU 2. 

$$
\begin{aligned}
p_1 &= \text{softmax}([q k_1^\top, q k_2^\top]) [v_1; v_2] \\
&= \frac{1}{z_1}[\exp(q k_1^\top), \exp(q k_2^\top)] [v_1; v_2] \\
z_1 &= \exp(q k_1^\top) + \exp(q k_2^\top)
\end{aligned}
$$

So on GPU1, if we do attention, we will get our partial result $o_1$

Our objective to caluculate $o$, which is defined by:

$$
\begin{aligned}
p &= \frac{1}{z}[\exp(q k_1^\top), \exp(q k_2^\top), \exp(q k_3^\top), \exp(q k_4^\top)] [v_1; v_2; v_3; v_4] \\
z &= \exp(q k_1^\top) + \exp(q k_2^\top) + \exp(q k_3^\top) + \exp(q k_4^\top)
\end{aligned}
$$

If we ignore the normalization factor, our desired result looks very like the summation of $p_1$ and $p_2$. But we can renormalize our partial result. You can check that

$$
p = \frac{z_1}{z}p_1 + \frac{z_2}{z}p_2
$$

This basically means that we can do distributed attention calculation on GPU1 and GPU2 to get partial results $p_1$ and $p_2$ and do some communication to get the ultimate result $p$. 

The ring-attention algorithm is built upon this powerful insight. We so far pretend that we have a full copy of $q$ on each rank, but in reality, $q$ is also sharded like $k$ and $v$. The following graph represent how the algorithm go from iteration 1 to iteration 2.

![RingAttentionRotation](/img/ring-attention-rotation.png)

$P_{ij}$ means the partial result calculated from ith query chunk and jth kv chunk. The $\text{sum}^*$ arrow in the graph is renormalized summation $\frac{z_1}{z}p_1 + \frac{z_2}{z}p_2$ that we showed above. On iteration 1, since KV1 and Q1 are on GPU1, KV2 and Q2 are on GPU2, etc, we get P11 on GPU1, P22 on GPU2, etc. On iteration 2, we send KV1 to GPU2, KV2 to GPU3, etc. This allowed us to calculate P21, P32, P43, and P14 and update our partial result. We do 3 such iterations, then we are going to have the result of full attention for each chunk on each GPU. But KV1 is on GPU4, KV2 is on GPU1, etc. So we need to do another iteration of communication so that the KV chunks go back to where they are started at.

The full ring-attention algorithm is nicely illustrated in this animation.

![KV-rotate](/img/KV-rotate.gif)

Query chunks remains on each GPU, while key-value chunks rotate through those GPU, hence the name ring-attention. 

Everytimes when we send a KV chunk to another GPU, we bears a communication cost of $4 \frac{h_{size}}{P}$ for each GPU. Since we need to do $P$ iterations, the communication cost for each GPU is $4 h_{size}$. So ring attention have a higher communication overhead than ulysses. However, a nice thing about ring-attention is that we can hide the communication with compute. While you are sending KV1 to GPU2, you can do the attention computation to get P11. So in principle, when the next iteration begins, you already get all the KV chunks ready and you start computation immediately. In theory, communication is free if you hide communication well. This and the fact that ring-attention is not limited by the number of attention heads makes ring-attention potentially more scalable than ulysses despite larger communication cost. Also, ring attention's P2P communication is less strict about the network topology. In the end, the biggest disadvantage of ring attention is its complexity.

## Context Parallelism

Context parallelism (CP) is basically the application of ring-attention in the inference scenario. It's often used in long-context summarization and long reasoning. A critical difference for inference is that inference has kvcache and has decoding stage. In decoding stage, we just need to compute the new token's relationship with all the past context, so the size of the query are much smaller than kvcache. Online serving is usually latency sensitive, so the number of queries in a batch is usually very small. Comparatively, the kvcache grows as the sequence grows. So it make no sense in the long sequence but small batch scenario to rotate kv while making queries fixed. 

In fact the author of Context Parallelism for Scalable Milion-Token Inference argues that in the long sequence decoding scenario, we should rotate query while keeping the kv fixed. This is called Ring Pass-Q. The algorithm for rotating KVCache in inference is called Ring Pass-KV.