---
layout: post
title: "Muon: Steepest Descent on Matrices, and Does It Kill Neurons?"
date: 2026-06-01
categories: [machine-learning, optimization]
---

# Muon: Steepest Descent on Matrices, and Does It Kill Neurons?

*Muon has quietly climbed to the top of the nanoGPT speedrun leaderboard and is now used in frontier-scale training runs. This post explains the mathematics that makes it tick, surveys the recent claim that it causes neuron death in MLP layers, and then presents my own small-scale encoder experiment that complicates that narrative.*

---

## Why Not Adam?

Adam is the default optimizer for almost everything. Its core idea is simple: maintain a per-parameter exponential moving average of squared gradients, and use it to rescale each coordinate before the update. Concretely, for a scalar parameter $w$ with gradient $g$:

$$
w \leftarrow w - \eta \frac{g}{\sqrt{v} + \epsilon}, \quad v \leftarrow \beta_2 v + (1 - \beta_2) g^2.
$$

This gives each coordinate an *adaptive learning rate* proportional to the inverse of its historical gradient RMS. The update direction is therefore $-\text{sign}(g)$ in the limit, which is steepest descent under the $L^\infty$ norm per-coordinate.

This works well when parameters are semantically independent scalars — biases, embedding tables, scalar gates. But most of the parameters in a transformer are *weight matrices*: the attention projections $W_Q, W_K, W_V, W_O$ and the MLP projections $W_\text{gate}, W_\text{up}, W_\text{down}$. A weight matrix $W \in \mathbb{R}^{m \times n}$ acts on *vectors*, and its natural "size" is not the sum of its entry magnitudes but rather its effect on vector norms. Adam treats each entry of $W$ independently, completely ignoring this matrix structure.

The question Muon answers is: *what is the steepest descent direction for a matrix parameter when we use a norm that respects its matrix nature?*

---

## Steepest Descent Under the Spectral Norm

### What should we measure: parameter change or activation change?

When we apply an update $\Delta W$ to a weight matrix $W \in \mathbb{R}^{m \times n}$, the downstream effect on an activation $h = Wx$ is:

$$
\Delta h = \Delta W \, x.
$$

The Frobenius norm $\|\Delta W\|_F$ measures the total magnitude of parameter change across all entries — a purely parameter-space notion. But from the perspective of what the network *does*, what matters is how much the activations shift for a given input. The worst-case activation shift over all unit-norm inputs is exactly the **spectral norm**:

$$
\max_{\|x\|=1} \|\Delta W \, x\| = \|\Delta W\|_2.
$$

This is the fundamental asymmetry between Frobenius and spectral norm: Frobenius counts parameter-space movement, spectral norm bounds functional change. If we want a fair "budget" for how much a single update step is allowed to disturb the network's function, the spectral norm is the right measure. Constraining $\|\Delta W\|_2 \leq \eta$ means that *no unit-norm input vector can have its activation shifted by more than $\eta$*, regardless of which direction you probe.

Frobenius norm by contrast has no such guarantee — a large Frobenius update can spread its mass across many nearly-redundant directions and barely affect any output, while a small but focused Frobenius update can shift activations dramatically.

### The Optimization Problem

This motivates framing one step of optimization as: find the direction $U$ that most decreases the loss while keeping the worst-case activation shift bounded:

$$
U^* = \arg\max_{U} \, \text{Tr}(G^\top U) \quad \text{subject to} \quad \|U\|_2 \leq 1,
$$

where $G = \nabla_W \mathcal{L}$ and the inner product $\text{Tr}(G^\top U)$ is the first-order decrease in loss. This is steepest descent in spectral norm.

**The solution is the polar factor.** Write $G = U_G \Sigma_G V_G^\top$ (thin SVD). Then:

$$
\text{Tr}(G^\top U) = \text{Tr}(V_G \Sigma_G U_G^\top U).
$$

Since $\|U\|_2 \leq 1$, every singular value of $U$ is at most 1. The trace inner product is maximized when we set every singular value of $U$ to 1 and align its singular vectors with those of $G$ — giving:

$$
U^* = \operatorname{polar}(G) = U_G V_G^\top.
$$

This is the semi-orthogonal matrix closest to $G$ in Frobenius norm, with **spectral norm exactly 1**. The unit spectral norm is not just a convenience: it means every step moves the network's function by the same worst-case amount, making the learning rate $\eta$ a true, interpretable bound on per-step activation shift.

Intuitively, $\operatorname{polar}(G)$ strips the singular values from $G$ — the "how much each direction matters" — and keeps only $U_G V_G^\top$ — the "which directions to move." All singular directions of the gradient are updated equally, rather than letting the largest singular value dominate (as it would under a plain gradient step).

Su Jianlin's blog post [苏剑林, 2024](https://www.spaces.ac.cn/archives/11647) gives a complementary derivation of this from the perspective of operator norms, showing that polar decomposition is the unique solution and connecting it to why spectral-norm-constrained descent outperforms Frobenius-norm descent (Adam) on matrix parameters.

---

## The Algorithm

A naive implementation would compute the SVD at every step, which costs $O(\min(m,n) \cdot mn)$ and is slow on modern accelerators. Muon instead uses a **Newton-Schulz iteration** to compute the polar factor using only matrix multiplications.

The quintic iteration from modded-nanoGPT is:

$$
X_{k+1} = \frac{1}{8}\left(15 X_k - 10 X_k X_k^\top X_k + 3 X_k (X_k^\top X_k)^2\right),
$$

initialized from a rescaled gradient $X_0 = G / \|G\|_F$. After 5–6 steps this converges to the polar factor with high precision. The only operations are matrix multiplications — no eigendecompositions, no square roots — making it hardware-efficient and easy to fuse.

The full Muon update for a matrix parameter $W$ is:

$$
\begin{aligned}
M_t &\leftarrow \beta_1 M_{t-1} + (1 - \beta_1) G_t \\
O_t &\leftarrow \operatorname{polar}(M_t) \\
W_{t+1} &\leftarrow W_t - \eta \lambda W_t - \eta \, O_t
\end{aligned}
$$

where $M_t$ is the gradient momentum buffer. The momentum makes the polar factor computation smoother and more stable, since $M_t$ averages over recent gradients before orthogonalization.

**Spectral vs. Frobenius.** Note that the update $O_t$ has spectral norm exactly 1 and Frobenius norm $\sqrt{\min(m,n)}$. Each step moves $W$ by the same amount regardless of how large or small the recent gradients were — only their *direction* matters. This is a strong form of scale invariance.

---

## The Neuron Death Claim

Tilde Research's Aurora paper [(Dewulf et al., 2026)](https://blog.tilderesearch.com/blog/aurora) raised a sharp concern: **Muon can cause persistent neuron death in MLP layers**, and the mechanism is geometrically inevitable for tall weight matrices.

### Leverage Scores

For a matrix $M \in \mathbb{R}^{m \times n}$ with thin SVD $M = U_r \Sigma V^\top$, define the **leverage score** of row $i$ as:

$$
\ell_i(M) = \|(U_r)_{i,\cdot}\|_2^2.
$$

The $m$ leverage scores sum to $n$ (since $\|U_r\|_F^2 = n$), and the row norm of the polar factor $\operatorname{polar}(M)$ at row $i$ equals $\sqrt{\ell_i(M)}$. A row with high leverage score gets a large update; a row with low leverage score gets a tiny update.

**For tall matrices ($m > n$)**, orthogonality imposes only that $U_r^\top U_r = I_n$, which means the $m$ leverage scores must sum to $n$ but are otherwise unconstrained — they can concentrate arbitrarily. This is in contrast to **square or wide matrices**, where $U_r$ is itself square and orthogonal, forcing $\ell_i = 1$ for all $i$.

### The Feedback Loop

Here is the self-reinforcing failure mode the Aurora authors identify:

1. Early in training, some MLP neurons receive smaller gradient rows than others — perhaps due to initialization or early data statistics.
2. The momentum buffer $M_t$ develops **non-uniform row norms**. Rows with consistently small gradients accumulate small momentum.
3. Via the leverage bound $\ell_i(M) \leq \|M_{i,\cdot}\|^2 / \sigma_n(M)^2$ (Claim 2 in the paper), rows with small momentum get small leverage scores in the polar factor $O_t$.
4. These rows receive near-zero weight updates. Their corresponding neurons in the MLP (specifically the up-projection and gate rows in a SwiGLU block) never learn.
5. Since they never update, their gradient signal remains small — reinforcing the small momentum — and the cycle continues.

The Aurora paper found this empirically at 340M and 1B scale: by step 500, more than one in four neurons in middle MLP layers showed near-zero leverage under Muon, with a sharply bimodal leverage distribution. Under their fix (U-NorMuon or Aurora), leverage remained isotropic throughout training.

**Why is this specific to tall matrices?** In the up-projection $W_\text{up} \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$, we have $d_\text{ff} \gg d_\text{model}$ (e.g., $4 \times$ expansion). This is a tall matrix. Its polar factor $UV^\top$ can have highly non-uniform row norms. By contrast, the down-projection $W_\text{down} \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$ is wide, and orthogonality forces uniform row norms automatically.

---

## My Experiment: No Dead Neurons at 4M Scale

The Aurora paper's evidence comes from large transformer pretraining (340M–1B parameters, billions of tokens). I ran Muon on a much smaller setting: a **4M-parameter encoder** trained on 4M samples, with SwiGLU MLPs, comparing AdamW vs. Muon.

### Methodology

A "dead" SwiGLU neuron requires *both* the gate row *and* the corresponding up row to show persistently low effective gradient and update norms — across all checkpoints after a reference step.

I tracked the **effective gradient norm ratio** and **effective update norm ratio** (each row's norm divided by the same-layer mean) across snapshots at steps 100, 150, 200, and 245. A neuron is declared dead if its ratio stays below a threshold for every snapshot after the reference step.

### Results

| Variant | Dead pairs @0.05× | Dead pairs @0.10× | Dead pairs @0.25× |
| --- | ---: | ---: | ---: |
| AdamW | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| Muon  | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

At the 0.25× threshold — already quite generous — neither optimizer produces a single dead neuron pair. At 0.5×, Muon shows 3 dead pairs (0.004%), versus 0 for AdamW, but this is negligible.

The row-level statistics tell a more nuanced story:

| Variant | Threshold | Low grad rows | Low update rows | Low both | Dead pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 0.25 | 74 (0.050%) | 0 (0.000%) | 0 (0.000%) | 0 |
| Muon  | 0.25 | 175 (0.119%) | 44 (0.030%) | 39 (0.026%) | 0 |

Muon does show a heavier persistent low-update tail than AdamW at the row level, which is consistent with the leverage anisotropy mechanism. But the paired gate-up death — the genuine neuron death — does not materialize. The window-max quantiles show that even the weakest rows maintain some update activity:

| Variant | Metric | q0.1% | q1% | q5% | Median |
| --- | --- | ---: | ---: | ---: | ---: |
| Muon | update window-max | 0.3432 | 0.5551 | 0.7147 | 1.0470 |
| AdamW | update window-max | 0.6709 | 0.8037 | 0.9167 | 1.2460 |

Even at the 0.1th percentile, the worst Muon row has a window-max update ratio of 0.34× — low, but not zero, and not persistently suppressed in the way the Aurora paper describes for large models.

### What Does This Mean?

The Aurora paper identified a *scale-dependent* failure mode. Their experiments with 340M and 1B parameter models trained for thousands of steps show the feedback loop crystallizing early and persisting. My 4M experiment, with far fewer parameters and a short 245-step window, likely does not give the feedback loop time or capacity to fully develop.

Several factors may explain the discrepancy:

**Training depth.** The self-reinforcing cycle needs time. The Aurora authors observe collapse as early as step 500, but their models trained for 10,000+ steps. With 245 steps I may simply be measuring before the pathology appears.

**Model width.** The Aurora paper notes that Aurora's gains *scale with MLP width*. A 4M encoder has much narrower MLPs than a 340M transformer; the leverage score distribution has fewer degrees of freedom to develop anisotropy.

**Initialization and conditioning.** At small scale, gradient norms may stay better-conditioned throughout training. The bound $\ell_i(M) \leq \|M_{i,\cdot}\|^2 / \sigma_n(M)^2$ tightens when $\sigma_n(M)$ (the smallest singular value of the momentum buffer) is large — which is more likely for smaller, better-conditioned models.

**The Muon leverage signal is there, just not catastrophic.** The 39 rows with persistently low gradient *and* low update norm (0.026%) hint that the mechanism is present but suppressed. At larger scale, this trace likely amplifies into the 25%+ dead-neuron regime the Aurora authors observe.

---

## Discussion

Muon's mathematical foundation is principled and elegant: it is steepest descent in spectral norm, the natural matrix operator norm. By stripping scale from the gradient and using only its directional information (the polar factor), Muon treats all singular directions of the gradient equally — unlike Adam, which privileges high-variance directions.

The neuron death concern from Aurora is real and theoretically grounded. For tall matrices, the polar factor can carry significant row-norm anisotropy, and the Muon momentum loop has no mechanism to correct it. Aurora's fix — adding a uniform row-norm constraint to the steepest-descent problem — is mathematically principled and yields a cleaner optimization problem on the joint Stiefel/equal-leverage manifold $\mathcal{M} = \{U : U^\top U = I,\, \text{diag}(UU^\top) = \frac{n}{m}\mathbf{1}\}$.

But my experiment suggests this pathology is not universal across scales. At 4M parameters and a short training window, Muon shows no meaningful neuron death — and in fact trains comparably to AdamW. This matters practically: if you are training small-to-medium models, vanilla Muon appears safe.

The responsible conclusion is scale-dependent: Muon is a theoretically sound improvement over Adam for matrix parameters, and its leverage-score failure mode is real but emerges primarily at scale. For large-scale pretraining — frontier models with wide MLPs and long training runs — Aurora or U-NorMuon are worth the modest overhead. For smaller encoder training, standard Muon holds up fine.

---

## References

- Kosson et al., "Muon: Momentum Orthogonalized by Newton-Schulz" (2024)  
- Dewulf\*, Pai\*, Yang\*, Zhang\*, Keigwin, "Aurora: A Leverage-Aware Optimizer for Rectangular Matrices," Tilde Research (2026). [blog](https://blog.tilderesearch.com/blog/aurora)  
- Su Jianlin, "Muon 优化器的数学原理" (2024). [spaces.ac.cn](https://www.spaces.ac.cn/archives/11647)  
- Jordan Keller, "modded-nanoGPT" speedrun. [GitHub](https://github.com/KellerJordan/modded-nanoGPT)
