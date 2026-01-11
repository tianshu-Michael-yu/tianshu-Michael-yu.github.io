---
layout: post
title: "Tiny Recursive Models (TRM): Recursive Reasoning with Tiny Networks"
date: 2026-01-10
categories: [machine-learning, reasoning, deep-learning]
---

## Tiny Recursive Models

**Tiny Recursive Models (TRM)** show that *recursive reasoning* plus *deep supervision* can outperform much larger models on hard reasoning tasks (Sudoku-Extreme, Maze-Hard, ARC-AGI) using **only ~5–7M parameters**.  
Instead of scaling depth or width, TRM **reuses a tiny network many times**, progressively improving its own solution.

![TRM](/img/trm.png)

---

## Motivation: Why Recursion Instead of Bigger Models?

Large language models struggle on tasks where:
- A single error invalidates the solution (Sudoku, ARC)
- Step-by-step correctness matters more than fluency
- Data is scarce (≈1k training examples)

Scaling parameters increases capacity but also **overfitting**.  
TRM takes a different route:

> **Fix the capacity, scale the reasoning.**

---

## Core Idea: Iterative Self-Correction

TRM maintains **two states**:

- **$y$** – the current *proposed solution* (embedded)
- **$z$** – a *latent reasoning state* (analogous to an internal chain-of-thought)

Given an input problem $x$, TRM repeatedly:
1. Refines its latent reasoning $z$
2. Uses $z$ to refine the solution $y$

This allows the model to **correct earlier mistakes** instead of committing to a single forward pass.

---

## Architecture Overview

TRM uses **a single tiny network** $f_\theta$ (typically a 2-layer Transformer).



---

## Algorithm (Pseudocode)

The following pseudocode makes the structure of **Tiny Recursive Models (TRM)** explicit: a *latent recursion* phase that updates the reasoning state, followed by an *answer refinement* step, wrapped inside a deep-recursion training loop.

### 1) Latent recursion + answer refinement

```python
# x : embedded input
# y : current embedded solution
# z : latent reasoning state
# f_theta : the same tiny network used everywhere

def latent_recursion(x, y, z, n=6):
    # Reasoning phase: update latent state z
    for _ in range(n):
        z = f_theta(x, y, z)

    # Decision phase: update solution y once
    y = f_theta(y, z)
    return y, z
```

### 2) Deep recursion wrapper (T rounds)

```python
def deep_recursion(x, y, z, n=6, T=3):
    # T-1 rounds without gradients (cheap iterative reasoning)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(x, y, z, n)

    # Final round with gradients (used for learning)
    y, z = latent_recursion(x, y, z, n)
    return y, z
```

### 3) Deep supervision training loop

```python
# N_sup : maximum supervision steps (e.g., 16)
# g     : output head mapping y -> logits
# h     : halting head mapping y -> halting logit

for (x_input, y_true) in dataloader:
    x = embed(x_input)
    y, z = init_y(), init_z()

    for step in range(N_sup):
        y, z = deep_recursion(x, y, z, n=6, T=3)

        logits = g(y)
        q      = h(y)

        # prediction + halting losses
        loss = CE(logits, y_true) + BCE(sigmoid(q), is_correct(logits, y_true))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # early stopping if the model believes it is correct
        if sigmoid(q) > 0.5:
            break
```

**Notes**
- The *same* network $f_\theta$ is reused everywhere; passing $(x, y, z)$ versus $(y, z)$ determines whether the model updates reasoning or refines the answer.
- Using $T−1$ rounds without gradients plus one round with gradients keeps memory usage low while still training multi-step improvement.

---

## Inference Structure

At inference time, TRM runs the same recursive update logic **without any losses or backpropagation**. The model repeatedly refines its solution using fixed inner recursion and adaptive outer stopping.

```python
# x : embedded input
# y : current embedded solution
# z : latent reasoning state
# f_theta : shared tiny network
# K_max : maximum number of improvement steps

def trm_inference(x, y, z, n=6, T=3, K_max=16):
    for k in range(K_max):                     # outer improvement loop
        for t in range(T):                     # deep recursion
            for _ in range(n):                 # latent reasoning
                z = f_theta(x, y, z)
            y = f_theta(y, z)                  # answer refinement

        # optional halting (same criterion as training)
        q = h(y)                               # halting logit
        if q > 0:                              # equivalent to sigmoid(q) > 0.5
            break

    return y
```

**Key points**
- The model performs `n × T` latent-reasoning updates per outer improvement step.
- The solution `y` is refined **T times per improvement step**, and up to `K_max` improvement steps overall.
- Deep supervision is a **training-only** mechanism; inference simply reuses the learned iterative solver.
- Halting is checked only at the outer loop, even though `y` is updated inside the inner loops.

---

## Mathematical Formulation

### State Variables

Let:
- $x ∈ ℝ^{L×D}$ : embedded input
- $y_t ∈ ℝ^{L×D}$ : solution embedding at step $t$
- $z_t ∈ ℝ^{L×D}$ : latent reasoning state

---

### Latent Recursion (Reasoning Phase)

For $n$ reasoning steps:

$$
z_{t,k+1} = f_θ(x, y_t, z_{t,k}), \quad k = 0 \dots n-1
$$

We denote the final latent state as:

$$
z_{t+1} = z_{t,n}
$$

This phase updates *how* the model reasons without yet changing the answer.

---

### Answer Update (Decision Phase)

Using the refined latent state:

$$
y_{t+1} = f_θ(y_t, z_{t+1})
$$

The same network $f_θ$ is reused; the input structure defines the role.

---

### Deep Recursion

To control memory cost, TRM performs:

- $T−1$ recursion rounds **without gradients**
- 1 recursion round **with gradients**

$$
(y_{t+1}, z_{t+1}) =
\mathcal{R}^{(T)}(x, y_t, z_t)
$$

Only the final recursion participates in backpropagation.

---

## Training Objective

### Prediction Loss

The solution embedding is decoded via an output head:

$$
\hat{y}_t = \arg\max (W y_t)
$$

$$
\mathcal{L}_{pred} = \mathrm{CE}(\hat{y}_t, y^*)
$$

---

### Halting Loss (Adaptive Computation)

TRM learns when to stop iterating.

Let $q_t$ be a halting logit:

$$
\mathcal{L}_{halt} =
\mathrm{BCE}(\sigma(q_t), \mathbf{1}[\hat{y}_t = y^*])
$$

Training halts early when the model predicts correctness.

---

### Total Loss

$$
\mathcal{L} = \mathcal{L}_{pred} + \mathcal{L}_{halt}
$$

---

## Why Two States ($y$ and $z$)?

- $y$ stores *what* the current solution is
- $z$ stores *how* the model reasoned to reach it

Empirically:
- One state → worse performance
- More than two states → worse generalization
- Exactly two states → best results

---


## Why Tiny Networks Work Better

Counterintuitively, **2-layer models outperform deeper ones**.

Reason:
- Data is scarce
- Larger models overfit
- Recursion provides effective depth

Effective depth is approximately:

$$
\text{Depth} ≈ T × (n+1) × \text{layers}
$$

Example:
- 2 layers × 42 recursions ≈ 84-layer behavior

---

## Observation: Reasoning Capacity vs Knowledge Capacity

The empirical result that *smaller* TRMs generalize better on ARC-AGI, Sudoku, and Maze should not be interpreted as evidence that reasoning capacity is inherently small or that larger models are fundamentally worse at reasoning.

Rather, it reflects two properties of the experimental setting:

1. **The tasks require almost no world knowledge.**  
   ARC-AGI and related puzzles are algorithmic and symbolic. They do not require linguistic understanding, factual recall, or common-sense priors about the real world. As a result, increasing parameter count mainly adds unused or overfittable capacity.

2. **The training regime is extremely data-limited.**  
   With only on the order of \(10^3\) training tasks, larger models are prone to shortcut learning and memorization rather than learning a robust iterative procedure.

From this perspective, TRM’s key contribution is architectural rather than parametric: it **decouples reasoning capacity from parameter count**. Reasoning depth is provided by *iteration over latent state*, while parameters primarily encode inductive biases and task-specific representations.

This suggests a more general interpretation:

- **Parameters control knowledge breadth** (what the model knows).
- **Recursion controls reasoning depth** (how long and how carefully the model can think).

Under this view, the observation that “smaller models work better” is specific to low-knowledge puzzle domains. In settings that require substantial world knowledge, variable-length outputs, or open-ended semantics, larger TRMs may benefit from increased parameterization while retaining efficient recursive reasoning.

---

## Comparison with Chain-of-Thought (CoT)

It is instructive to contrast TRM’s recursive reasoning with **Chain-of-Thought (CoT)** reasoning in large language models. Both aim to improve reasoning by “thinking longer,” but they scale computation and memory in fundamentally different ways.

### How CoT scales reasoning

In CoT-based models, reasoning depth is increased by generating longer token sequences. This has several consequences:

- **Compute scales quadratically** with reasoning length due to self-attention over longer sequences.
- **Memory scales linearly** with the number of generated tokens, since all intermediate tokens must be stored.
- Reasoning is **entangled with generation**: each reasoning step is a committed token and cannot be revised.
- Errors early in the chain can irreversibly corrupt the final answer.

In practice, increasing reasoning via CoT incurs rapidly increasing compute and memory costs.

### How TRM scales reasoning

TRM increases reasoning depth through **iteration over a fixed-size latent state**:

- **Compute scales linearly** with the number of recursion steps.
- **Memory remains constant**, since the latent state has fixed dimensionality and is repeatedly updated.
- Reasoning is **non-committal**: intermediate latent states can be revised and corrected.
- The final answer is produced only after sufficient refinement.

This makes TRM closer to an **iterative solver** than a generative process.

### Key contrast

| Aspect | Chain-of-Thought | TRM |
|------|------------------|-----|
| Reasoning medium | Token sequence | Fixed-size latent state |
| Compute vs reasoning | Quadratic | Linear |
| Memory vs reasoning | Linear | Constant |
| Error correction | Difficult | Natural |
| Reasoning vs knowledge | Entangled | Decoupled |

From this perspective, TRM can be viewed as an **architectural alternative to CoT**: instead of scaling reasoning by growing sequences, it scales reasoning by reusing computation over a persistent internal state.

This distinction helps explain why TRM is particularly effective in low-data, algorithmic domains, and why its recursive structure may remain advantageous even as models are scaled to handle broader, knowledge-intensive tasks.

---

## Training Details

- Optimizer: AdamW (β₁=0.9, β₂=0.95)
- Learning rate: 1e−4 (embeddings: 1e−2)
- Hidden size: 512
- Layers: 2
- Recursions: $n=6$, $T=3$
- Max supervision steps: $N_{sup}=16$
- Exponential Moving Average (EMA): 0.999

---

## Results Snapshot

TRM achieves strong generalization with orders-of-magnitude fewer parameters than LLMs:

![ARC-AGI-1](/img/arc-agi-1.png)
![ARC-AGI-2](/img/arc-agi-2-trm.png)

- **87%** on Sudoku-Extreme
- **85%** on Maze-Hard
- **44.6%** on ARC-AGI-1
- **7.8%** on ARC-AGI-2

Using only **~7M parameters**.

---

## Key Takeaways

1. Reasoning depth can replace parameter count
2. Recursion is a powerful regularizer
3. Deep supervision enables stable iterative improvement
4. Simplicity beats biologically motivated complexity

---

## Open Questions

- Why does recursion generalize better than depth?
- Can TRM be extended to generative modeling?
- What are the scaling laws for recursion vs parameters?

---

## Autoregressive TRM with Per-Token Refinement

I want to propose a possible extension of TRM that is **not explored in the original paper**: turning the three-time-scale TRM into an **autoregressive model** by wrapping it inside an outer token-generation loop.

In this proposal, TRM would no longer be a one-shot predictor, but instead act as an **adaptive compute engine per token**.

### High-level idea

- The inner TRM performs **iterative refinement** to produce a *single next-token distribution*.
- An outer autoregressive loop emits one token at a time, appends it to the input context, and repeats until a stopping token is generated.

This allows variable-length outputs while preserving TRM’s recursive reasoning structure.

### Per-token refinement with adaptive compute

For each decoding step:

- The prefix tokens are fixed, and their key–value (K,V) cache is reused.
- TRM refines the latent solution state corresponding to the **current token** over multiple recursion steps.
- Each refinement step updates the token’s representation, and therefore its K,V, while attending to the cached prefix.
- Once the halting condition is met, the final refined state is used to emit the token, and its K,V are committed to the cache.

Intermediate refinement steps are **not serialized into tokens** and remain fully revisable.

### Compute and memory characteristics

Under this design:

- **Compute per refinement step scales linearly** with prefix length, since attention reuses cached K,V from previous tokens.
- **Total compute per token scales linearly** with the number of refinement steps.
- **Memory usage remains bounded** by the KV cache for emitted tokens plus the (small) model weights.
- Model capacity can be shifted from parameters to **recursion depth**, improving the balance between memory and compute.

This contrasts with Chain-of-Thought reasoning, where increasing reasoning length directly increases sequence length, attention cost, and memory usage.

### Why this is appealing

This autoregressive TRM formulation combines several desirable properties:

- **Adaptive compute**: easy tokens halt early, hard tokens receive more refinement.
- **Decoupling of knowledge and reasoning**: parameters store knowledge, recursion provides reasoning depth.
- **Inference efficiency**: reduced weight memory pressure and better utilization of available compute.
- **Edge deployment**: smaller models with dynamic compute are well suited to memory-constrained devices.

Conceptually, this shifts reasoning from *sequence expansion* to *latent refinement*, suggesting a path toward language models that are more efficient, more controllable at inference time, and less dominated by static parameter count.

---

*Based on “Less is More: Recursive Reasoning with Tiny Networks” (Jolicoeur‑Martineau, 2025).*
