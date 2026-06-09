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

## Three Characteristics of a Well-Behaved Model

Before asking what the ideal update looks like, it helps to ask what it means for a model to be *well-behaved* throughout training. Su Jianlin's framework ([苏剑林, 2025](https://www.spaces.ac.cn/archives/11340)) identifies three stability properties for a linear layer $W \in \mathbb{R}^{m \times n}$:

**1. Forward stability.** The output scale should match the input scale. For any input $x$:

$$
\|Wx\|_2 \leq \|W\|_2 \cdot \|x\|_2,
$$

so bounding $\|W\|_2 = O(1)$ ensures activations don't blow up through depth. The tightest such bound is exactly the spectral norm, achieved when $x$ is the top right singular vector of $W$.

**2. Backward stability.** Gradients $\delta$ propagate backwards through $W^\top$:

$$
\|W^\top \delta\|_2 \leq \|W^\top\|_2 \cdot \|\delta\|_2 = \|W\|_2 \cdot \|\delta\|_2.
$$

Since $\|W^\top\|_2 = \|W\|_2$, forward and backward stability are controlled by the same spectral norm. A model that is forward-stable is automatically backward-stable.

**3. Increment stability.** When $W$ is updated to $W + \Delta W$, the change in output for any input $x$ is:

$$
\|(W + \Delta W)x - Wx\|_2 = \|\Delta W \, x\|_2 \leq \|\Delta W\|_2 \cdot \|x\|_2.
$$

For each training step to make a comparably-sized functional change across all layers — regardless of model width or gradient magnitude — we need $\|\Delta W\|_2 = O(\eta)$ uniformly.

The **spectral condition** unifies all three: $\|W\|_2 = O(1)$ at initialization (properties 1 & 2), and $\|\Delta W\|_2 = O(\eta)$ at each step (property 3). Muon enforces the increment condition exactly — the polar factor $\operatorname{polar}(G)$ satisfies $\|\operatorname{polar}(G)\|_2 = 1$, so every step produces a worst-case functional shift of exactly $\eta$, independent of layer width or gradient scale. Weight decay handles the parameter condition, keeping $\|W\|_2 \leq 1/\lambda$ throughout training. Together they maintain all three stability properties.

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

where $G = \nabla_W \mathcal{L}$. This is steepest descent in spectral norm.

**Why $\text{Tr}(G^\top U)$?** The gradient $G = \nabla_W \mathcal{L}$ is the matrix of partial derivatives, with $G_{ij} = \partial \mathcal{L}/\partial W_{ij}$. When the weight matrix shifts by a small amount $\delta W$, the first-order change in loss is the sum of each partial derivative times its corresponding weight change:

$$
\delta \mathcal{L} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial W_{ij}} \delta W_{ij} = \sum_{i,j} G_{ij} \, \delta W_{ij}.
$$

This entry-wise sum is the **Frobenius inner product** $\langle G, \delta W \rangle_F$. To see why this equals a trace, expand:

$$
\text{Tr}(G^\top \delta W) = \sum_i (G^\top \delta W)_{ii} = \sum_i \sum_j G^\top_{ij} \, \delta W_{ji} = \sum_{i,j} G_{ji} \, \delta W_{ji} = \sum_{i,j} G_{ij} \, \delta W_{ij}.
$$

So $\delta \mathcal{L} = \text{Tr}(G^\top \delta W)$ — the loss change is just the "dot product" of $G$ and $\delta W$ when both are treated as flat vectors. The first-order Taylor expansion for a step $\delta W = -\eta U$ is therefore:

$$
\mathcal{L}(W - \eta U) \approx \mathcal{L}(W) - \eta \, \text{Tr}(G^\top U).
$$

So $\text{Tr}(G^\top U)$ measures how much the loss decreases per unit of $U$ in the gradient's direction. Maximizing it over $\|U\|_2 \leq 1$ asks: *among all matrix updates with spectral norm at most 1, which one points most in the gradient's direction?*

Crucially, the **constraint** is spectral ($\|U\|_2 \leq 1$) while the **objective** uses the Frobenius inner product. These are different norms doing different jobs: the Frobenius inner product measures alignment with the gradient; the spectral norm constraint controls functional impact on activations. By norm duality — the dual of the spectral norm is the nuclear norm — the maximum value of $\text{Tr}(G^\top U)$ over the spectral-norm ball is $\|G\|_*$ (the nuclear norm of $G$), and the maximizer is the polar factor $U_G V_G^\top$.

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

## Muon Variants: The Schatten Norm Family

The spectral, Frobenius, and nuclear norms are all special cases of the **Schatten $p$-norm**:

$$
\|A\|_p = \left(\sum_i \sigma_i(A)^p\right)^{1/p},
$$

with $p=\infty$ giving the spectral norm $\sigma_{\max}$, $p=2$ giving the Frobenius norm $\sqrt{\sum \sigma_i^2}$, and $p=1$ giving the nuclear norm $\sum \sigma_i$. The key duality: the dual norm of $\|\cdot\|_p$ is $\|\cdot\|_q$ for $1/p + 1/q = 1$, so spectral and nuclear are duals of each other, and Frobenius is self-dual.

Since steepest descent under $\|\cdot\|$ maximizes $\text{Tr}(G^\top U)$ over $\|U\| \leq 1$, and the maximizer is the dual-norm ball's exposure in direction $G$, we get a distinct optimizer for each norm:

| Budget constraint | Update $U^*$ | Frobenius norm of $U^*$ | Name |
|---|---|---|---|
| $\|U\|_2 \leq 1$ (spectral) | $U_G V_G^\top$ — all $\sigma_i = 1$ | $\sqrt{\min(m,n)}$ | **Muon** |
| $\|U\|_F \leq 1$ (Frobenius) | $G / \|G\|_F$ — normalized gradient | $1$ | Normalized SGD |
| $\|U\|_* \leq 1$ (nuclear) | $u_1 v_1^\top$ — top singular pair only | $1$ | Rank-1 power step |

The differences become sharp when you look at **how each update treats the singular value spectrum of the gradient**:

- **Frobenius** renormalizes the whole gradient to unit length. It keeps the relative sizes of all singular values intact — directions with large $\sigma_i$ get large updates, small-$\sigma_i$ directions get starved. This is just rescaled gradient descent; it does nothing to equalize learning across the weight matrix's singular directions.

- **Spectral (Muon)** sets *every* singular value of the update to 1. The polar factor $U_G V_G^\top$ discards the singular values entirely, giving equal update energy to every singular direction of the gradient. This is the equalizing property: no direction is privileged just because it happened to accumulate a large gradient.

- **Nuclear** takes only the top singular pair $u_1 v_1^\top$ — a rank-1 step in the direction of greatest gradient. This concentrates all the update mass on the single most important direction and ignores the rest. It is the most "focused" option but also the most aggressive in ignoring lower-ranked gradient directions.

### The Frobenius Norm as the Wrong Penalty for Matrices

The reason Frobenius is the "wrong" choice for matrix parameters connects back to the activation-change argument. Suppose you normalize the gradient to $\|G\|_F = 1$. The Frobenius norm spreads its mass across all $mn$ entries — but the functional effect on activations is dominated by the large singular values. A gradient with $\sigma_1 \gg \sigma_2, \ldots$ and $\|G\|_F = 1$ will drive $\|Gx\|$ almost entirely via $\sigma_1$; the remaining directions receive tiny updates. Over many steps, the weight matrix's lower singular directions learn much more slowly, effectively wasting capacity.

Spectral-norm budgeting avoids this: by forcing all singular values of the update to 1, every singular direction of the gradient gets equal learning opportunity regardless of the raw gradient magnitude in that direction. This is why the polar factor has Frobenius norm $\sqrt{\min(m,n)}$ rather than 1 — the extra mass comes from the equalization, not from aggressive scaling.

More precisely: if the gradient spectrum is well-distributed (all $\sigma_i \approx \sigma$), then $\|G\|_F \approx \sigma \sqrt{\min(m,n)}$ and $G/\|G\|_F \approx U_G V_G^\top / \sqrt{\min(m,n)}$, so Frobenius-normalized SGD approximates a scaled Muon. The gap widens as the gradient becomes rank-deficient or ill-conditioned — exactly the regime common in deep network training.

### How Muon Controls the Spectral Norm

The polar factor $O_t = \operatorname{polar}(M_t)$ always has **spectral norm exactly 1**: all its singular values are forced to 1 by construction. Each Muon step therefore adds a unit-spectral-norm matrix to $W$. Without weight decay, this would let $\|W\|_2$ grow without bound — every step pushes the weight matrix by a fixed-magnitude operator, so the spectral norm grows at rate $\sim \eta$ per step with no ceiling.

Weight decay is what provides the ceiling. The update rule is:

$$
W_{t+1} = W_t + \eta(O_t - \lambda W_t) = (1 - \eta\lambda)W_t + \eta O_t.
$$

Because $\|O_t\|_2 \leq 1$, whenever $\|\lambda W_t\|_2 > 1$ (i.e., $\|W_t\|_2 > 1/\lambda$), the weight decay term $-\eta\lambda W_t$ is larger in operator norm than the update $\eta O_t$, so $\|W_{t+1}\|_2 < \|W_t\|_2$ — the spectral norm shrinks back. More precisely (Chen et al., [2025](https://arxiv.org/abs/2506.15054), Proposition 1): for any update $X_{t+1} = X_t + \eta_t(O_t - \lambda X_t)$ with $\|O_t\|_2 \leq 1$ and $\eta_t \lambda \leq 1$,

$$
\|X_t\|_2 - \frac{1}{\lambda} \leq \prod_{s=0}^{t-1}(1 - \eta_s\lambda)\!\left(\|X_0\|_2 - \frac{1}{\lambda}\right).
$$

Since $(1-\eta_s\lambda) < 1$, the excess above $1/\lambda$ decays **exponentially** to zero, regardless of the initialization. So $\|W\|_2$ converges to the constraint ball $\|W\|_2 \leq 1/\lambda$ at an exponential rate determined by $\eta\lambda$.

### Why Weight Decay is Necessary in Muon

Setting $\lambda = 0$ collapses the spectral norm constraint: the bound $1/\lambda \to \infty$, so no constraint is enforced and $\|W\|_2$ can grow unboundedly.

Without weight decay the update is $W_{t+1} = W_t + \eta O_t$ with $\|O_t\|_2 = 1$, so the spectral norm satisfies $\|W_t\|_2 \leq \|W_0\|_2 + \eta t$ — it grows linearly. In practice this leads to training instability: the weight matrices become large operators that amplify activations at every layer, destabilizing the forward pass and making loss spikes likely.

The $\lambda$-dependence also means **weight decay is not just regularization** in Muon — it is a core hyperparameter that sets the scale of the network's linear maps. Choosing $\lambda$ is equivalent to choosing the maximum allowed spectral norm $1/\lambda$ of each weight matrix, which in turn sets the maximum activation amplification per layer. This is a much more interpretable handle than the ad-hoc weight decay schedules common in AdamW training.

### Practical Variants

Su Jianlin's optimizer guide [苏剑林, 2025](https://www.spaces.ac.cn/archives/11416) covers the practical implications of these norm choices and how to use Muon alongside other optimizers in a real training setup. The key design question for practitioners is: **which parameters get which norm?**

In the standard MuonAdam setup, weight matrices get the spectral-norm LMO (polar factor), while vector parameters (biases, LayerNorm scales, embedding tables) get Adam — because Adam's per-coordinate adaptive scaling is the natural steepest descent under a per-entry $\ell^\infty$ norm, which is appropriate for semantically independent scalar parameters. Crawshaw et al. ([2025](https://arxiv.org/abs/2510.09827)) formalize this as constrained steepest descent under the product norm:

$$
\|(W_1, \ldots, W_L, \theta)\|_\text{MuonAdam} = \max\!\left(\max_\ell \|W_\ell\|_2,\; \frac{\eta_m}{\eta_b} \|\theta\|_{\text{ada},\infty}\right),
$$

where $\|\cdot\|_{\text{ada},\infty}$ is Adam's adaptive $\ell^\infty$ norm. The ratio $\eta_m/\eta_b$ allows separate learning rates for matrices and vectors, which in practice is crucial — Muon matrices typically use $10\times$ larger learning rates than the Adam parameters.

Their variant **MuonMax** uses regularized (rather than constrained) steepest descent and scales each layer's polar factor by the nuclear norm of its momentum buffer, giving each matrix an adaptive step size. This is more expensive (requires the nuclear norm at each layer) but significantly more robust to learning rate tuning.

### Why Embeddings and LM Head Use Normalized SGD

MuonAdam applies Muon to matrix parameters and Adam to scalars/vectors. But there is a third category that needs its own treatment: **embedding tables** and the **LM head**. These are matrices, but applying Muon to them is the wrong choice — and the reason comes down to the structure of their inputs ([苏剑林, 2025](https://www.spaces.ac.cn/archives/11647)).

For a regular linear layer $W \in \mathbb{R}^{m \times n}$, the input $x \in \mathbb{R}^n$ is a dense continuous vector. The worst-case output shift from an update $\Delta W$ over all unit-norm inputs is the spectral norm:

$$\max_{\|x\|_2 = 1} \|\Delta W \, x\|_2 = \|\Delta W\|_2.$$

This is exactly why spectral norm is the right budget for linear layers, and why the polar factor (which fixes $\|\Delta W\|_2 = 1$) gives increment stability.

For an **embedding table** $E \in \mathbb{R}^{V \times d}$, the input is always a one-hot vector $e_i$ for some token $i$. The embedding lookup is a row read:

$$(\Delta E)\, e_i = \Delta E[i, :].$$

The functional effect of any update on a given input touches only a single row. The worst-case shift over all possible inputs is therefore the **max row norm**:

$$\max_i \|\Delta E[i, :]\|_2 = \|\Delta E\|_{2,\infty}.$$

Since inputs are never dense, the spectral norm is the wrong budget — it measures the worst case over all unit-norm directions, most of which never occur. The right increment stability condition is $\|\Delta E\|_{2,\infty} \leq \eta$: every token embedding moves by at most $\eta$.

Plugging this into the steepest descent problem:

$$U^* = \arg\max_U \, \text{Tr}(G^\top U) \quad \text{s.t.} \quad \max_i \|U[i,:]\|_2 \leq 1.$$

The constraint decouples across rows, so we can maximize each row's contribution independently. By Cauchy–Schwarz, $G[i,:] \cdot U[i,:] \leq \|G[i,:]\|_2 \|U[i,:]\|_2 \leq \|G[i,:]\|_2$, with equality when $U[i,:] = G[i,:] / \|G[i,:]\|_2$. The solution is:

$$U^*[i,:] = \frac{G[i,:]}{\|G[i,:]\|_2} \quad \text{for each row } i.$$

This is **normalized SGD**: normalize each row of the gradient to unit norm before the update step. It is the per-row analog of Muon — instead of orthogonalizing across all singular directions (appropriate when the input spans all of $\mathbb{R}^n$), it normalizes independently in each row (appropriate when each input activates exactly one row).

The **LM head** $W_\text{out} \in \mathbb{R}^{V \times d}$ follows the same logic from the output side. Each row $W_\text{out}[i,:]$ controls the logit for token $i$ via $W_\text{out}[i,:] \cdot h$. The per-token logit shift from a row update is $\Delta W_\text{out}[i,:] \cdot h$, bounded by the row norm. With $V \gg d$, the LM head is a tall matrix — exactly the regime where Muon's polar factor can produce highly non-uniform row norms (the leverage-score failure mode from the Aurora analysis), leaving some tokens' representations nearly frozen. Row-wise normalization avoids this by giving every token an equal per-step update.

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
- Su Jianlin, "MuP之上：1. 好模型的三个特征" (2025). [spaces.ac.cn](https://www.spaces.ac.cn/archives/11340)  
- Su Jianlin, "MuP之上：2. 线性层与最速下降" (2025). [spaces.ac.cn](https://www.spaces.ac.cn/archives/11605)  
- Su Jianlin, "MuP之上：3. 特殊情况特殊处理" (2025). [spaces.ac.cn](https://www.spaces.ac.cn/archives/11647)  
- Su Jianlin, "Muon续集：为什么我们选择尝试Muon？" (2025). [spaces.ac.cn](https://www.spaces.ac.cn/archives/10739)  
- Chen et al., "Muon Optimizes Under Spectral Norm Constraints" (2025). [arXiv:2506.15054](https://arxiv.org/abs/2506.15054)  
- Jordan Keller, "modded-nanoGPT" speedrun. [GitHub](https://github.com/KellerJordan/modded-nanoGPT)
