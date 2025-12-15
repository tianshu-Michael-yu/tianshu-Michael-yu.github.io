# Continuous Autoregressive Language Models

In our previous post, [Sequence Parallelism]({% post_url 2025-11-07-sequence-parallelism %}), we treated *sequence length* as a fixed property of the training batch, and asked how to shard computation when a single sequence becomes too long to process efficiently. That discussion focused on parallelizing attention under strict dependency constraints.

This post explores a complementary axis: **reducing the number of autoregressive steps themselves**. Continuous Autoregressive Language Models (CALM) do this by increasing the *semantic bandwidth* of each step, replacing next-token prediction with **next-vector prediction**, where a single continuous vector represents a chunk of $K$ discrete tokens.

---

## From next-token to next-vector

Let a token sequence be

$$
X = (x_1, x_2, \dots, x_T), \quad x_t \in \mathcal{V}.
$$

We choose a chunk size $K$ and group the sequence into

$$
S = T / K
$$

non-overlapping chunks (assume divisibility for simplicity). CALM introduces an encoder that maps each chunk of $K$ tokens into a vector in $\mathbb{R}^\ell$, yielding a shorter sequence

$$
Z = (z_1, z_2, \dots, z_S), \qquad
z_i = f_{\text{enc}}(x_{(i-1)K+1}, \dots, x_{iK}).
$$

The autoregressive factorization becomes

$$
p(Z) = \prod_{i=1}^{S} p(z_i \mid z_{<i}).
$$

Structurally this looks identical to a standard language model, but the output space is now continuous and uncountable. As a result, the usual “linear layer + softmax over vocabulary” construction no longer applies, and the conditional density $p(z_i \mid z_{<i})$ is intractable. CALM therefore treats the model as **implicit**: it is trained and evaluated through sampling rather than likelihoods.

---

## The chunk autoencoder

At the heart of CALM is a lightweight autoencoder that learns an almost-bijective mapping between

- a chunk of $K$ discrete tokens $x_{1:K} \in \mathcal{V}^K$, and
- a continuous latent vector $z \in \mathbb{R}^\ell$.

Formally,

$$
f_{\text{enc}} : \mathcal{V}^K \rightarrow \mathbb{R}^\ell, \qquad
g_{\text{dec}} : \mathbb{R}^\ell \rightarrow \mathcal{V}^K,
$$

with reconstruction $g_{\text{dec}}(f_{\text{enc}}(x_{1:K})) \approx x_{1:K}$.

For simplicity and efficiency, the autoencoder is **context-free**: each chunk is encoded independently of surrounding chunks.

### Architecture

A common (and CALM-compatible) way to implement the autoencoder is:

**Encoder**
1. Embed each token in the chunk: $e_j = E[x_j] \in \mathbb{R}^d$ for $j=1,\dots,K$.
2. Apply a position-wise MLP to each: $\tilde{e}_j = \text{MLP}(e_j)$.
3. Flatten and compress: $\tilde{e} = [\tilde{e}_1;\dots;\tilde{e}_K] \in \mathbb{R}^{Kd}$, then $u = W_c \tilde{e} \in \mathbb{R}^{d}$.
4. Project to latent: $z = W_z \, \phi(u)$ for some nonlinearity $\phi$.

**Decoder**
1. Lift: $u' = W'_z z$.
2. Expand back to $K$ slots: $\hat{e} = W_e \, \phi(u') \in \mathbb{R}^{Kd}$ and reshape into $(\hat{e}_1,\dots,\hat{e}_K)$.
3. Token logits: $\ell_j = E^\top \hat{e}_j$ (tying input/output embeddings).
4. Tokens: $\hat{x}_j = \arg\max \text{softmax}(\ell_j)$ (or sample).

### Reconstruction loss

The basic objective is standard cross-entropy over the $K$ positions:

$$
\mathcal{L}_{\text{ae}}(x_{1:K})
=
- \sum_{i=1}^{K}
\log p_{\text{dec}}\!\left(x_i \mid z = f_{\text{enc}}(x_{1:K})\right).
$$

With small $K$ (e.g. $K=4$), very high reconstruction accuracy can be achieved with surprisingly small $\ell$.

---

## Making the latent space robust

A reconstruction-only autoencoder learns a brittle latent space: infinitesimal perturbations in $z$ can decode to unrelated token chunks. This is catastrophic for autoregressive modeling, where prediction errors are unavoidable. CALM therefore explicitly regularizes the latent space.

### Variational regularization

Instead of outputting a single point, the encoder produces parameters of a diagonal Gaussian:

$$
(\mu, \sigma) = f_{\text{enc}}(x_{1:K}), \qquad
z \sim \mathcal{N}(\mu, \sigma^2 I).
$$

The training objective becomes

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{ae}}
+
\beta \, \mathcal{L}_{\text{KL}},
$$

where

$$
\mathcal{L}_{\text{KL}}
=
- \frac{1}{2}
\sum_{i=1}^{\ell}
\left(1 + \log \sigma_i^2 - \sigma_i^2 - \mu_i^2 \right).
$$

This discourages arbitrarily precise encodings and encourages a smooth latent manifold.

### KL clipping

To prevent posterior collapse (unused latent dimensions reverting to pure noise), apply per-dimension KL clipping:

$$
\mathcal{L}^{\text{clip}}_{\text{KL}}
=
\sum_{i=1}^{\ell}
\max\!\left(\lambda_{\text{KL}}, \mathcal{L}_{\text{KL},i}\right).
$$

This ensures every latent dimension carries useful signal.

### Dropout-induced redundancy

Two practical regularizers during autoencoder training:

- dropout on the latent vector $z$ before decoding,
- random masking of input tokens within a chunk.

Together, these force the representation to be redundant and tolerant to perturbations—exactly what the downstream continuous language model needs.

---

## The CALM autoregressive loop

CALM combines three components:

1. a Transformer backbone,
2. a lightweight generative head that samples $z_i$,
3. the frozen autoencoder decoder that maps $z_i$ back to $K$ tokens.

At step $i$:

1. The previous $K$ tokens are embedded and compressed into a single vector.
2. The Transformer produces a hidden state $h_{i-1}$.
3. A stochastic generative head samples $z_i \in \mathbb{R}^\ell$.
4. The decoder reconstructs the next $K$ discrete tokens.

Although the model predicts vectors, the **input remains discrete tokens**, which empirically stabilizes training.

---

## The energy-based generative head

The generative head is designed for **single-step continuous sampling**. It takes

- the Transformer hidden state $h_{i-1} \in \mathbb{R}^d$,
- a noise vector $\varepsilon \sim U[-0.5, 0.5]^{d_{\text{noise}}}$,

and outputs $z_i$.

A convenient formulation is:

$$
e_0 = W_\varepsilon \varepsilon, \qquad
c = W_h h_{i-1},
$$

followed by $B$ residual refinement blocks

$$
e_{b+1} = e_b + \text{MLP}_b([e_b, c]),
$$

and a final projection

$$
z_i = W_{\text{out}} e_B.
$$

---

## Training without likelihoods: the energy loss

Because $p(z_i \mid z_{<i})$ is intractable, CALM trains the generative head using a **strictly proper scoring rule**, specifically the *energy score*.

For a predictive distribution $P$ and observation $y$, the energy score is

$$
S(P, y)
=
\mathbb{E}_{x', x'' \sim P}\!\left[\|x' - x''\|^\alpha\right]
-
2 \, \mathbb{E}_{x \sim P}\!\left[\|x - y\|^\alpha\right],
\qquad \alpha \in (0, 2).
$$

In practice, expectations are approximated by Monte Carlo sampling. Let

- $\tilde{z}_{i,1}, \dots, \tilde{z}_{i,N}$ be samples from the model,
- $z_{i,1}, \dots, z_{i,M}$ be samples from the encoder’s Gaussian posterior.

A common estimator (using $\alpha=1$) is:

$$
\mathcal{L}_{\text{energy}}
=
\sum_{i=1}^{S}
\left(
\frac{2}{NM}
\sum_{n=1}^{N}
\sum_{m=1}^{M}
\|\tilde{z}_{i,n} - z_{i,m}\|
-
\frac{1}{N(N-1)}
\sum_{n \neq k}
\|\tilde{z}_{i,n} - \tilde{z}_{i,k}\|
\right).
$$

The first term pulls model samples toward the target distribution; the second enforces diversity and prevents collapse.

---

## Why this can speed things up

Standard autoregressive models generate **one token per step**. CALM generates **$K$ tokens per step**.

Conceptually:

- effective sequence length drops from $T$ to $T/K$,
- attention and other sequence-length-dependent costs shrink accordingly.

Sequence parallelism helps when $T$ is fixed and large; CALM attacks the problem earlier by changing what a “step” represents.

---

## Summary

CALM reframes language modeling around three ideas:

1. **High-bandwidth steps**: predict continuous vectors instead of low-information tokens.
2. **Robust latent spaces**: use variational regularization, KL clipping, and dropout so small errors remain decodable.
3. **Likelihood-free training**: replace cross-entropy with strictly proper scoring rules that only require sampling.

The result is a model that remains autoregressive in structure, but changes the unit of generation from *token* to *vector*.
