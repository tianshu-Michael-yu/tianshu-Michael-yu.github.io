
# Model Parallelism

Model parallelism shards the model 

### Tensor Parallelism (TP)
Let our model be $Y=XA$. We can split our models along its columns $A = [A_1, A_2]$. Then

$$
[Y_1, Y_2] = [XA_1, XA_2]
$$

Because of this we can do the computation $Y_1=XA_1$ and $Y_2=XA_2$ on two gpus. We split the same weight tensor onto two GPUs hence the name tensor parallelism.
TP cut latency, and ideally it cut latency into half, since you are doing the same computation A in parallel on two GPUS.

### Pipeline Parallelism (PP)

Let our model have two stages A, B, meaning the model do $ Z=XAB $. We can split the computation to two steps.

$$
Y=XA \\
Z=YB
$$

We can compute Y on GPU0, send Y to GPU1 and compute Z on GPU1. Since when GPU1 is doing computation for Z, GPU0 is idle. We can add another input to keep the pipeline busy. Below is an illustration of the pipeline, where $X_1, X_2, X_3$ are just different inputs. 

|Time| 0       | 1      | 2      | 3      |
|-----|:------:|:------:|:------:|:------:|
|GPU 1|        | $Y_1B$ | $Y_2B$ | $Y_3B$ |
|GPU 0| $X_1A$ | $X_2A$ | $X_3A$ |        |

PP doesn't cut latency, it even slightly degrades it. Originally you just do the operation of A and B in series on the same device, now you just do them serially on different device with the added latency of interdevice communication. 

### Expert Parallelism (EP)

![Alt text](/img/moe_layer.png)

Expert is a more specialized type of parallelism. They can only be applied for Mixture of Experts (MoE) models. The moe model route a token to its corresponding expert (in practice, maybe several experts). You can think the expert is just a linear layer for now (in practice, usually an mlp layer). You do the normal computation and send back the token. The key is that a MoE layer have many experts $A_1, A_2, ..., A_n$. But each token will only activate a subset of the expert. So not all parameters used in the computation of a token. In expert parallelism, we place each expert on a different GPU (or more practically, several experts on a GPU). For instance, if your token activate parameter $A_l$, only the GPU hosting that parameter is active, while all other GPU are idle. So we usually schedule a batch of tokens and hoping that those tokens are distributed to different experts evenly that each GPU process the same amount of compute. 

Whether expert parallelism cuts latency is more complicated. If the token needs to activate multiple parameters and those parameters on different GPU, it could cut the latency. But if all activated parameter for that token is on the same GPU, then no latency benefit. So load balancing is important for EP.

## Application of model parallelism in inference and serving

Real models are not simple linear layers. It has complicated structures.

![Alt text](/img/tensor_parallelism.png)

The above is how TP partition the self-attention and MLP blocks. PP simply partitons model along transformer blocks. (A transformer block is consist of self-attention and MLP blocks.) In real TP, each transformers blocks require two all reduces operation, while in PP, each transformers blocks require two send-recv operation (except the first block and the last block).

Let the hidden_state (Z) size to be $h_{size}$. Let a model's TP size is n. This means we shard the model using the above TP method into n shards. One all-reduce's communication cost per gpu (assuming RingAllreduce Algorithm) is $4(n-1)\frac{h_{size}}{n}$. Let the model's PP size to be m. One send/recv communication kernel's cost per is $h_{size}$. So per kernel wise, TP's communication operator consumes four times more communication bandwith than PP. Also, the total amount of all-reduce in a tranformers is proportional to the number of layers, while the number of send/recv only depends on the PP size. Let a transformer have $k$ layers. The total amount of communication cost per GPU for TP is $8k(n-1)\frac{h_{size}}{n} \approx 8kh_{size}$, while the total amount of communication cost per GPU for PP is $(m-1)h_{size}$. Most model have more than $28$ layers, while it's rare we need PP size more than 16. So the communication cost of TP dwarfs that of PP. Because within a node, we can leverage the high bandwith communication method NVLink, we usually only do TP within a node and do PP across nodes with the less impressive RDMA connection. 

![Alt text](/img/intra-and_inter-layer_parallelism.png)

As we mentioned above, PP can't cut per token latency, so PP can't help much in the decode stage in improving latency. However, PP can help improve the time to first token (TTFT) latency. 

![Alt text](/img/PPvsCPP.png)

As the picture showed, this is because that in the prefill stage, we already know all the tokens in the context. We can schedule all tokens in the context to the pipeline without having to wait for the processing of the previous token. This helps the prefilling stage to finish much faster hence reducation in TTFT.

In online serving, we usually disaggregate prefill and decode. On the machine which we purely run prefill, PP is a very useful strategy.

EP on the other hand is tricky. We will just touch on one thing about EP here. 

![Alt text](/img/multi_node_ep.webp)

EP uses this all-to-all communication to route tokens from different Data Parallel (DP) rank to different experts and after computation, use another all-to-all to route those token back to their respective DP rank. The communication cost for an all-to-all per GPU is $\frac{top_kh_{size}}{l}$, assuming there're $l$ EP rank, where $top_k$ is the number of experts each token will be routed to. So ideally, as EP sizes grows, the communication volume per GPU actually goes down. In practice, all to all is notoriously hard to be implemented efficiently. PP's send/recv are not synchronized ops for the entire PP group, while every member in the ep group need to wait for all peers to send their shard. If one rank is slow, everyone is slowed down. It's hard for EP to be load balanced, because it's perfectly possible that all token will be routed to one expert, which exceberate the problem.  

### More things to consider in actual serving

A lot of model, such as Qwen3, uses Grouped Query Attention (GQA). In GQA, each kv heads are shared by a group of q heads. So usually the number of kv heads are very small. In Qwen3-235B-A22B, the num of kv_heads are only 4. So in this case, if you do TP=8, you must duplicate kv heads and their corresponding kv cache on each GPU. This will waste both time and compute. So if you care about throughput, we usually restrict our tensor parallel size to 4.

Let's take Qwen3-235B-A22B as an example. Since one H100 GPU has 80GB of memory and the model weight is `bfloat16`, if we use 8 H100 to serve, we only have $80-235 \times 2 / 8 = 21.25 GB$, which is not enough for kv cache. So we usually serves using 2 node. Each node have 8 H100. Based on the above discussion we typically use TP=4, PP=4. This gives 16 shards of the model that can be fit on all of the 16 H100 on the 2 node.