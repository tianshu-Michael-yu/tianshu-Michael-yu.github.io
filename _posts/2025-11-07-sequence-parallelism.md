# Sequence Parallelism

In our previous post, [Model Parallelism]({% post_url 2025-11-07-model-parallelism %}), we introduced model parallelism. It shards the size of the model. Since in online serving, the number of tokens in a batch is small, model weight dominates the memory consumption. Model parallelism is more common in serving. In training, however, the amount of data in a batch can be much larger than the weight of the model, so we need to shard the data. There're two dimensions when we parallelize the data. Parallelize across different sequence and parallelization within sequence. The previous one is data parallelism and the latter one is sequence parallelism. We will discuss sequence parallelism in this post.

## Attention

![Attention](/img/attention_compute_graph.png)

## Ulysses

![Ulysses](/img/ulysses_compute_graph.png)

## RingAttention
![KVRotate](/img/KV-rotate.gif)