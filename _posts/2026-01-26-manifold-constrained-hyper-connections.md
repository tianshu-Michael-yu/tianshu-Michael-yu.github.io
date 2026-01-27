# Manifold-Constrained Hyper-Connections

The residual connection from ResNet has remained largely unchanged for a decade. The pattern is simple: add the layer output back to the input, $x_{l+1} = x_l + F(x_l, W_l)$. This identity mapping lets gradients flow unimpeded through deep networks and has become a fundamental building block of transformers and LLMs.

Hyper-Connections (HC) extended this paradigm by widening the residual stream and diversifying connection patterns. Instead of a single residual path, HC maintains $n$ parallel streams and uses learnable matrices to route information among them. The performance gains are substantial, but the approach has a hidden cost: it breaks the identity mapping property that made residual connections stable in the first place.

This post looks at Manifold-Constrained Hyper-Connections (mHC), which restores stability by projecting the learnable mappings onto doubly stochastic matrices.

## The Problem with Hyper-Connections

In standard residual networks, recursively unrolling the connection across layers gives:

$$
x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i).
$$

The term $x_l$ passes through unchanged—this is the identity mapping. Signal magnitude stays bounded because we are just summing residuals onto a stable baseline.

HC replaces the scalar identity with a learnable matrix $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$:

$$
x_{l+1} = \mathcal{H}_l^{\text{res}} x_l + \mathcal{H}_l^{\text{post} \top} F(\mathcal{H}_l^{\text{pre}} x_l, W_l),
$$

where $x_l \in \mathbb{R}^{n \times C}$ is the widened residual stream. Unrolling this across layers produces a composite mapping $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$ in front of the initial features. Nothing constrains this product to behave like an identity, so signals can explode or vanish as they propagate through the network.

The DeepSeek team measured this instability directly. They define the *Amax Gain Magnitude* as the maximum absolute row sum (forward signal gain) or column sum (backward gradient gain) of the composite mapping. For a well-behaved identity mapping this value should stay near 1. In their 27B model trained with HC, the Amax Gain Magnitude spiked to nearly 3000—a clear sign of exploding residual streams.

## Doubly Stochastic Matrices

The fix is to constrain $\mathcal{H}_l^{\text{res}}$ to be a doubly stochastic matrix: a non-negative matrix whose rows and columns each sum to 1. Formally, mHC projects the residual mapping onto the Birkhoff polytope:

$$
\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}}) = \left\{ \mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_l^{\text{res}} \mathbf{1}_n = \mathbf{1}_n,\; \mathbf{1}_n^\top \mathcal{H}_l^{\text{res}} = \mathbf{1}_n^\top,\; \mathcal{H}_l^{\text{res}} \geq 0 \right\}.
$$

Why does this help? Doubly stochastic matrices have several nice properties for training stability:

1. **Norm preservation.** The spectral norm of a doubly stochastic matrix is bounded by 1, so the mapping is non-expansive and cannot blow up gradients.
2. **Closure under multiplication.** The product of doubly stochastic matrices is itself doubly stochastic. This means the composite mapping across many layers inherits the same stability guarantee.
3. **Convex combination of permutations.** Geometrically, every doubly stochastic matrix is a weighted average of permutation matrices (Birkhoff–von Neumann theorem). The residual mapping therefore acts as a soft permutation that mixes features across streams without amplifying or attenuating them.

When $n = 1$, the only doubly stochastic "matrix" is the scalar 1, recovering the original identity mapping.

## The Sinkhorn-Knopp Algorithm

mHC computes the learnable coefficients similarly to HC. Given the flattened hidden state $\vec{x}_l \in \mathbb{R}^{1 \times nC}$, we apply a linear projection and add a static bias:

$$
\tilde{\mathcal{H}}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \text{mat}(\vec{x}'_l \varphi_l^{\text{res}}) + \mathbf{b}_l^{\text{res}},
$$

where $\vec{x}'_l = \text{RMSNorm}(\vec{x}_l)$. The resulting matrix $\tilde{\mathcal{H}}_l^{\text{res}}$ is unconstrained, so we project it onto the Birkhoff polytope using the Sinkhorn-Knopp algorithm.

The algorithm is straightforward. Starting from $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\text{res}})$ (element-wise exponentiation ensures positivity), we alternate row and column normalization:

$$
\mathbf{M}^{(t)} = \mathcal{T}_r\left( \mathcal{T}_c(\mathbf{M}^{(t-1)}) \right),
$$

where $\mathcal{T}_r$ divides each row by its sum and $\mathcal{T}_c$ divides each column by its sum. As $t \to \infty$ the matrix converges to a doubly stochastic one. In practice, 20 iterations suffice.

For the pre and post mappings, mHC simply applies a sigmoid to enforce non-negativity:

$$
\mathcal{H}_l^{\text{pre}} = \sigma(\tilde{\mathcal{H}}_l^{\text{pre}}), \qquad \mathcal{H}_l^{\text{post}} = 2\sigma(\tilde{\mathcal{H}}_l^{\text{post}}).
$$

## System-Level Optimizations

Widening the residual stream by a factor of $n$ increases memory access costs proportionally. The original HC paper did not address this overhead, which limited practical scalability.

mHC introduces several infrastructure optimizations:

**Kernel fusion.** Computing the coefficients involves RMSNorm, linear projections, Sinkhorn-Knopp iterations, and the application of the mappings. mHC fuses these operations into a small number of custom CUDA kernels using TileLang. The RMSNorm division is reordered to follow the matrix multiplication, maintaining mathematical equivalence while reducing memory traffic.

**Recomputation.** Instead of storing all intermediate activations for the backward pass, mHC discards the mHC-specific activations after the forward pass and recomputes them on-the-fly during backpropagation. For a block of $L_r$ consecutive layers, only the input to the first layer needs to be stored. The optimal block size that minimizes total memory footprint is approximately:

$$
L_r^* \approx \sqrt{\frac{nL}{n+2}}.
$$

**DualPipe overlap.** In pipeline-parallel training, the $n$-stream residual incurs extra communication at stage boundaries. mHC extends the DualPipe schedule to overlap this communication with computation. The post-residual kernels for MLP layers run on a dedicated high-priority compute stream, and attention kernels avoid persistent mode so they can be preempted for communication.

With these optimizations, mHC adds only 6.7% training overhead at expansion rate $n = 4$.

## Results

On a 27B MoE model trained for 50k steps, mHC achieves a final loss reduction of 0.021 compared to the baseline. More importantly, the training is stable: gradient norms stay well-behaved throughout, whereas HC shows a loss spike around step 12k that correlates with unstable gradients.

| Benchmark | Baseline | HC | mHC |
|-----------|----------|-----|-----|
| BBH (EM) | 43.8 | 48.9 | **51.0** |
| DROP (F1) | 47.0 | 51.6 | **53.9** |
| GSM8K (EM) | 46.7 | 53.2 | **53.8** |
| MMLU (Acc) | 59.0 | 63.0 | **63.4** |

mHC consistently outperforms both the baseline and HC across reasoning and knowledge benchmarks. The stability analysis confirms the mechanism: the Amax Gain Magnitude of the composite mapping in mHC stays below 2, three orders of magnitude smaller than HC's spikes.

## Takeaways

- HC's performance gains come at the cost of breaking the identity mapping property, which leads to exploding/vanishing signals in deep networks.
- Constraining the residual mapping to be doubly stochastic restores stability while preserving the benefits of multi-stream residuals.
- The Sinkhorn-Knopp algorithm provides a differentiable projection onto the Birkhoff polytope.
- Careful infrastructure work (kernel fusion, recomputation, pipeline overlap) makes the approach practical at scale with minimal overhead.
