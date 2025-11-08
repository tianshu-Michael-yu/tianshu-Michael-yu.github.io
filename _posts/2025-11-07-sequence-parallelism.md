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

Here the inputs are the projections of the four tokens: a, b, c, d. For a, a0 is the projection of the 0th q head, a2 is the projection of 1th q head, etc. All the projections of a is on GPU 0, all the projections of b is on GPU 1. After the all-to-all, GPU0 gets b0, c0, d0 from GPU1, 2, 3. Similarly for other GPU. Instead of having all the projections of the same token, each GPU now haves all the tokens projected through the same q head. So the communication cost per GPU is $\frac{h_{size}}{P^2}(P-1) \approx \frac{h_{size}}{P}$. So the benefit of using ulysses is that as you increase the number of sequence parallel rank, the communication cost per GPU is going to decrease.

After all-to-all, on each GPU, ulysses can do full-attention for the entire sequence but only with some of the heads. In order to get the result of full-attention for the part of the sequence on each rank, the ulysses does the second all-to-all.

The advantage of ulysses lies in its implicity. As you will see, the algorithm for ring-attention is much more complex. Also, the communication overhead is relatively low. The disadvantages of ulysses are two fold. First, the degree of sequence parallel is limited by the amount of attention heads. Second, all-to-all communication is sensitive to latency and has certain requirements for network topology. 

## RingAttention

![KV-rotate](/img/KV-rotate.gif)


## Context Parallelism