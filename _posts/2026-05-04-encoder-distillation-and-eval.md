---
layout: post
title: "Distilling Vision Encoders from Multiple Teachers — and How to Tell If It Worked"
date: 2026-05-04
categories: [machine-learning, vision, distillation]
---

# Distilling Vision Encoders from Multiple Teachers — and How to Tell If It Worked

*A practitioner's view on multi-teacher distillation for general-purpose visual representations, and the evaluation suite that should follow it.*

---

## Why distill encoders at all

Modern vision encoders are increasingly *general-purpose*: a single ViT backbone is asked to do classification, retrieval, dense prediction, robotics control, and conditioning for generative models. The frontier models that produce the best representations — DINOv2, SigLIP/SigLIP 2, CLIP at scale, MAE/MAE-2, AM-RADIO, Theia, SAM's image encoder — each capture *complementary* signal. CLIP and SigLIP are strong at semantics and language alignment; DINOv2 produces the cleanest dense features for segmentation, depth, and correspondence; MAE preserves pixel-level reconstructive structure; SAM's encoder carries the strongest spatial priors for promptable mask quality.

No single pretraining objective dominates. So the natural move is to *amalgamate* them into one student that inherits the union of capabilities — at a fraction of the cost. That is the bet behind multi-teacher encoder distillation, and it is where AM-RADIO, Theia, UNIC, and a flurry of follow-ups live.

This post walks through the design choices that actually matter when you build such a student, and then — because distillation is uniquely easy to deceive yourself about — the evaluation methodology I'd insist on before believing any number.

## What multi-teacher encoder distillation actually optimizes

The high-level recipe is simple: feed the same image to N frozen teachers and to a student backbone, then push the student to match each teacher's output through a teacher-specific projection head.

$$
\mathcal{L} = \sum_{t=1}^{N} \lambda_t \, \mathcal{D}_t\!\left( h_t(f_\theta(x)),\; g_t(x) \right)
$$

Here $f_\theta$ is the shared student trunk, $h_t$ is a small per-teacher head (usually a 1–3 layer MLP or transformer), $g_t$ is teacher $t$'s frozen output, and $\mathcal{D}_t$ is the matching loss. Almost every interesting design choice is hidden in those four symbols.

**What to match.** Logits-style matching is mostly irrelevant here — these are encoders, not classifiers. You're choosing among (a) the global representation (CLS token / pooled embedding), (b) the full patch-token grid, and (c) intermediate features. Global-only matching is cheap but throws away exactly the signal that makes DINOv2 valuable: spatially coherent dense features. AM-RADIO's lesson is that *dense* matching at the patch level is what unlocks segmentation- and detection-grade students. If a teacher's value proposition is dense (DINOv2, SAM, MAE), match its tokens. If it's global (CLIP, SigLIP), match the summary embedding plus, optionally, a coarser version of its tokens.

**Resolution mismatch.** Teachers were trained at different input sizes and patch sizes (CLIP at 224 with patch 14 or 16, DINOv2 at 224 or 518 with patch 14, SigLIP 2 at 256+, SAM at 1024). The student is one tower with one patch size. You'll need either (i) per-teacher resize + interpolation of their token grid to the student's grid, or (ii) variable-resolution training where the student processes multiple crops at the teacher's native resolution. Option (ii) preserves dense signal better but multiplies compute. AM-RADIO uses partition-by-resolution mini-batches; Theia uses bilinear feature resizing; both work, but the bilinear shortcut tends to blur fine features on small objects.

**Distance function.** Cosine similarity in a normalized space is the safe default for global embeddings. For dense features, smooth-L1 or MSE on L2-normalized tokens beats raw MSE because it stops one teacher with large feature norms from dominating the gradient. Some authors add a Pearson/correlation term; in practice the gains are marginal once you normalize.

**Teacher weighting.** Hand-tuning $\lambda_t$ is the part nobody likes. Three approaches in use: fixed weights (simple, brittle), uncertainty-weighted losses à la Kendall et al. (one learnable log-variance per teacher), and gradient-norm balancing (GradNorm). The dirty secret is that fixed weights, with a short hyperparameter sweep on a downstream proxy, are usually within noise of the fancier schemes. What *does* matter is normalizing each teacher's loss to a comparable scale before you weight it; otherwise your weights are just absorbing scale differences.

**Heads vs. shared trunk.** Per-teacher heads are non-negotiable — the teachers live in different feature spaces, and forcing the student trunk to produce all of them at once collapses representation quality. The interesting question is *where* the heads attach. Attaching all heads to the final block makes the trunk a Swiss-army feature; attaching different heads to different depths (e.g., a SAM head on a mid-block, a CLIP head on the final block) tends to give better dense metrics but is finicky to tune.

**Token types and per-token losses.** A modern ViT trunk emits three kinds of tokens, and a careful distillation recipe treats them differently because they carry different signal:

- The **CLS token** (or a learned "summary" token) is what gets pooled into a single vector for global tasks. In CLIP/SigLIP it's the embedding the contrastive loss is applied to; in DINOv2 it's where most of the image-level semantics live. Its job is to summarize the image into one direction in feature space.
- **Patch tokens** are the per-location features on the $h \times w$ grid. They carry locality: object boundaries, stuff/things distinctions, spatial layout. These are the tokens that drive ADE20K dense probe, depth, correspondence, mask quality.
- **Register tokens** were introduced in *Vision Transformers Need Registers* (Darcet et al., 2024) and are now standard in DINOv2/v3 and AM-RADIO. They're a small set (e.g., 4–16) of extra learned tokens with no positional meaning. They act as a sink: without them, ViTs spontaneously co-opt a handful of *patch* tokens (often in low-information regions of the image) to store global "scratch" information, which corrupts the patch grid and produces those characteristic high-norm artefacts you see in DINOv2 attention maps. With registers, the artefact tokens move off the patch grid into the registers, and the patch grid stays clean.

The reason this matters for distillation is that **a single matching loss across all tokens is the wrong design**. AM-RADIO's recipe is the canonical example of separating them:

- *Summary loss* (CLS-to-CLS): cosine distance between the student's summary token (after a teacher-specific summary head) and the teacher's CLS / pooled output. This is the loss that transfers CLIP-style semantics or DINO's global concept structure. Cosine outperforms L1/MSE/Smooth-L1 here because the summary direction matters more than its magnitude.
- *Feature loss* (patch-to-patch): cosine + Smooth-L1 between the student's patch tokens (after a teacher-specific feature head) and the teacher's spatial token grid. This is the loss that transfers locality. Empirically, the AM-RADIO ablation found that *spatial* matching helped not just dense downstream tasks — it improved global quality too, which is the cleanest evidence that the "global summary" you get out of a contrastive teacher is leaving structure on the table that you only recover by matching the full grid.
- *Registers are not distilled directly, and — perhaps surprisingly — the RADIO student does not appear to use them at all.* The RADIOv2.5 paper describes the student as a plain ViT with cropped position embeddings and per-teacher MLP adaptors, with no register tokens in the student backbone, even though one of its teachers (`DINOv2-g-reg`) uses them. C-RADIOv4 doesn't mention student-side registers either; the only occurrence of "register" in the v4 tech report is a bibliographic citation. There is **no published ablation** comparing student-with-registers vs. student-without-registers in any RADIO paper. So treat the architectural choice as defensible-by-inference rather than empirically validated. The likely reasons it works without student registers: the *teachers* whose patches are matched (DINOv2-g-reg, DINOv3, SAM) have already absorbed their high-norm outliers into their own registers *before* exposing patch features as targets, so the student is being trained against an already-clean distribution; SigLIP/SigLIP 2 has no registers and no comparable artefact pattern (contrastive training doesn't push patches into outlier roles the way DINO-style local objectives can); and adding registers to the student would create tokens with no supervision signal at all — neither teacher's registers correspond to the student's, and a positional loss between unordered learned tokens is meaningless. C-RADIOv4 instead defends patch-grid cleanliness through a *shift-equivariant dense loss* (independent random crops for student and teacher, with shift-aligned matching) that prevents the student from baking position-fixed artefacts into specific patch locations. That's protecting the same property registers protect, but at the loss level rather than the architecture level. If you're building your own multi-teacher recipe, this is genuinely an open ablation worth running — there's no public answer for whether student-side registers would help on top of supervised, already-clean patch targets.

A subtler point: AM-RADIO ablated using a *separate* student CLS token vs. average-pooling the student's patch tokens for the summary loss. Average-pooling looks tempting (one fewer token), and it improves the summary loss in isolation — but it *hurts* the feature (patch) loss, because you're now leaning on the patch grid to do double duty. They picked separate CLS tokens for exactly this reason. The general lesson: give each loss its own dedicated tokens, or you get cross-talk where one objective degrades another's signal.

**Adapter / per-teacher head design — and why it matters even with one teacher.** I left this for last in the loss/architecture knobs because it's the one most people under-engineer. The per-teacher head $h_t$ is the only thing that can absorb the geometric mismatch between the student's feature space and teacher $t$'s — different dimensionality, different norm scale, different anisotropy. There are three common choices and they are not equivalent:

1. **Bare linear projection.** $h_t(z) = W_t z + b_t$. One matrix per teacher. Cheap, easy to interpret, and the natural starting point. The implicit assumption is that the student's trunk features and the teacher's targets are already related by an affine transform — which is almost never true after PHI-S, let alone before.
2. **Single MLP** (`Linear → GELU → Linear`, optionally with a LayerNorm). The default in many distillation codebases. Adds one nonlinearity, which is enough to fit *some* curvature but typically not enough when the teacher's manifold is highly non-affine relative to the student's (e.g., distilling DINO patch features from a CLIP-pretrained init, or matching a teacher whose features live on a different effective rank).
3. **The RADIO-style residual adapter (`MLP2` in `radio/adaptor_mlp.py`).** This is what NVlabs uses in v2.5/v4 and it is more elaborate than the usual MLP. The actual structure in the released code is:

   ```
   (optional pre-norm: LN + GELU)
     ↓
   Linear(in → hidden)                       # fc1, bare linear, no activation
     ↓
   num_inner × residual block:
       x = x + Linear(GELU(LayerNorm(x)))    # pre-LN, single Linear inside the residual
     ↓
   LayerNorm + GELU + Linear(hidden → out)   # "final" projection
   ```

   So it's: input projection into a hidden width → some small number of pre-LN GELU residual blocks (typically 2, but `num_inner` is data-driven from the checkpoint) → a pre-norm output projection. RADIO uses two independent instances per teacher — `head_mlp` for the summary token and `feat_mlp` for patch features — wired up in `GenericAdaptor`. The feature head additionally supports an optional spatial upsample.

   What this buys you over a single-MLP head: (i) the residual blocks let depth grow without making early training unstable, so the head can fit nontrivial curvature in the teacher manifold; (ii) the pre-LN inside each block stabilizes gradients flowing back into the trunk, which is what you actually care about — the head's job is to deliver a *clean* gradient to the backbone, not to be small; (iii) the final pre-norm + GELU + linear means the output isn't a raw linear projection from the residual stream, which empirically improves match quality at the tail of the distribution where teacher features have outliers.

**This matters even for single-teacher distillation.** I ran this myself: distilling from one teacher only, the RADIO-style residual adapter beat both a bare linear projection and a single-layer MLP on downstream metrics, by a margin large enough that it wasn't seed noise. The naive intuition — "with one teacher you don't need scale balancing, so a linear should be fine" — misses the point. The head isn't only there to balance teachers; it's there to absorb the curvature mismatch between *any* two feature spaces. A linear projection forces the student's trunk to do all the nonlinear bending itself, which both wastes trunk capacity on plumbing and produces worse downstream features because the trunk is now overfitting to one specific output geometry.

The practical rule I'd give: **use a residual adapter with at least two pre-LN blocks even when N = 1**. Hidden width = teacher feature dim is a fine default. The extra parameter count is negligible compared to the backbone, and the downstream gains hold up across teachers. If you want a single sentence rationale: the head should be expressive enough that the trunk can stay general-purpose; if the head is too weak, the trunk gets specialized as a teacher-shaped projection and you've lost the whole point of distillation.

**PHI-S: why teacher feature distributions need standardization.** Once you have two losses across N teachers, the next thing that bites you is that **teachers' activation statistics are wildly incompatible**. CLIP's CLS token might have feature variance ~1 with light tails; DINOv2's patch tokens have huge dynamic range with heavy-tailed outliers (the high-norm artefact tokens, even with registers, leave residual heavy tails); SAM's encoder lives at a different scale entirely. If you naively apply MSE or Smooth-L1, the teacher with the largest variance gets implicitly the largest weight in the gradient, regardless of whatever $\lambda_t$ you set. This is the "scale absorption" problem I flagged earlier under teacher weighting, but it's worse than it first sounds: variance differs *per dimension* of the feature vector, not just per teacher, so even per-teacher loss normalization isn't enough.

**PHI-S** ("PHI Standardization", from Heinrich et al., NVIDIA, 2024) is the cleanest fix and is now the default in the RADIO line. The idea, in three steps:

1. Estimate each teacher's per-feature mean and covariance over a sample of activations.
2. Whiten each teacher's distribution to zero mean and identity covariance — but a naïve PCA-style whitening produces dimensions with *different* scales after rotation back, which still lets some dimensions dominate.
3. Apply a **Hadamard matrix** rotation. The Hadamard matrix has the property that all its rows have unit $L_2$ norm and entries $\pm 1/\sqrt{d}$, so when you whiten and then rotate by $H$, every dimension of the resulting distribution has the *same* scale. This is *isotropic* standardization — each feature dimension contributes equally to MSE/Smooth-L1.

Concretely, for each teacher $t$ you precompute a fixed transform $T_t = H \Sigma_t^{-1/2}$ and a fixed mean $\mu_t$, then apply $\tilde{g}_t(x) = T_t (g_t(x) - \mu_t)$ before the loss. The student head is trained to predict $\tilde{g}_t(x)$, not $g_t(x)$. At inference time you can either invert the transform (if you need teacher-aligned features) or just use the standardized feature space directly.

Why this matters in practice:

- *Teachers become loss-balanced for free.* With PHI-S, you can use $\lambda_t = 1/N$ across teachers and get behavior that previously required GradNorm or learned uncertainty weights. The paper shows PHI-S beats MSE, Smooth-L1, and standard whitening across the board.
- *Per-teacher feature heads can be smaller.* When the target distribution is well-conditioned, the head doesn't have to absorb scale — so head-collapse pressure goes down.
- *It composes with the per-token-type losses above.* You apply PHI-S separately to each teacher's CLS distribution and each teacher's patch-token distribution, because their statistics are different.

PHI-S is the unsexy load-bearing piece of multi-teacher distillation: not glamorous, but the difference between a clean recipe and one where you spend weeks tuning loss weights and never quite figuring out why one teacher's signal is dominating.

**Data.** Pretraining data matters more than the loss. The teachers themselves were trained on huge image corpora; if you distill on a small or biased dataset, you bake that bias into the student even though the teachers wouldn't have. A web-scale unlabeled image set (LAION-style, DataComp filtered, or the DINOv2 LVD-142M curation) is the floor. Distillation is *extremely* sample-efficient compared to from-scratch pretraining — you typically need 10–30% of the original tokens to recover most of teacher quality — but "less data" is not the same as "tiny data."

## The single biggest failure mode

The thing that bites people is **target collapse at the head**. The student looks great on the distillation loss, the heads' cosine-with-teacher numbers approach 0.95, and downstream metrics are still mediocre. What's happening: the heads have absorbed all the per-teacher idiosyncrasy and the trunk is producing a generic feature that's *easy to project* but doesn't actually carry teacher-specific structure. You only see this if you evaluate the **trunk** directly, with frozen features and no head — which is exactly what the evaluation section below is built around.

## Optimizer geometry: choosing Muon, and what it actually does

Most distillation recipes default to AdamW out of habit. The optimizer choice meaningfully changes what the trunk learns, though, and once you understand the geometry, Muon turns out to be a clean fit for the two-loss, multi-teacher setting we've been describing. The next few subsections are the why; the practical recipe lives at the bottom for skimmers.

### Optimizers as choices of norm

Every gradient-based optimizer can be characterized as steepest descent under some norm constraint. Pick a norm $\|\cdot\|$, define your update as

$$\Delta W \;=\; \arg\max_{\|\Delta\|\,\le\,1}\;\langle -\nabla L,\;\Delta\rangle,$$

and the optimizer's behavior falls out. The inner product is the first-order predicted decrease in loss; the constraint sets the "budget" the parameter is allowed to move per step (the learning rate $\eta$ scales the answer back to whatever step size you want). All of an optimizer's personality lives in the choice of norm, because the norm determines the shape of the unit ball and therefore the shape of the maximizer.

Under a Frobenius ball you get vanilla SGD's direction. Under an $\ell_\infty$ ball on entries you get sign-SGD. Under a *coordinate-wise weighted* $\ell_\infty$ ball — entries normalized by a running second-moment estimate $\sqrt{\hat v}$ — you get the (approximate) AdamW step:

$$\Delta W_{ij} \;\approx\; \frac{-\hat m_{ij}}{\sqrt{\hat v_{ij}}}.$$

The Muon paper calls this the "Max-of-Max norm": you cap each entry independently. The construction treats a weight *matrix* as a bag of scalars and gives each scalar its own adaptive step size. That's a perfectly reasonable thing to do for embeddings, biases, and RMSNorm scales — parameters that genuinely are bags of scalars. It is *not* a reasonable thing to do for weight matrices that act as linear operators on activations, which is what almost all the parameters in a ViT trunk are.

### Spectral norm and the operator view

The natural norm for a weight matrix isn't entry-wise; it's operator-wise. A weight matrix $W \in \mathbb{R}^{m\times n}$ sits inside the network as a linear map $x \mapsto Wx$. What matters for stability is how much $W$'s action on its input changes when you update it — the spectral norm of the change $\|\Delta W\|_2$, not its entry-wise magnitude. Two updates can have identical Frobenius norms but very different operator behavior: a rank-1 update concentrates its full Frobenius budget along one singular direction and produces a big spectral change there, while a rank-$n$ update spreads the same budget across many directions for a smaller per-direction change. The Frobenius budget doesn't see this; the spectral budget does.

Concretely, the spectral norm of $A$ is its largest singular value:

$$\|A\|_2 \;=\; \sigma_{\max}(A) \;=\; \max_{\|x\|_2 = 1}\|Ax\|_2 \;=\; \sqrt{\lambda_{\max}(A^\top A)}.$$

In practice nobody computes a full SVD for this; a few iterations of power iteration on $A^\top A$ converge linearly at rate $(\sigma_2/\sigma_1)^2$, and spectral normalization in GANs uses one iteration per training step.

A confusion worth flagging while we're here: "the L2 norm of a matrix" is ambiguous. Treated as a flattened vector, it's the Frobenius norm $\sqrt{\sum_{ij} A_{ij}^2}$. Treated as the operator norm induced by the L2 vector norm, it's the spectral norm. They coincide only for rank-1 matrices; in general $\|A\|_2 \le \|A\|_F \le \sqrt{r}\,\|A\|_2$ where $r$ is the rank. PyTorch's `grad_norm` (the number you log to wandb) is the Frobenius variety, applied to the entire flattened parameter vector — not the spectral norm of any individual matrix. Whenever this section says "spectral norm," it's the operator one.

### Muon's update rule

Muon's whole substance is one step: given the EMA of past gradients $M_t$, replace it with its orthogonalized version before applying as an update. If $M_t = U\Sigma V^\top$ is the SVD, the orthogonalized update is

$$O_t \;=\; UV^\top.$$

That throws away $\Sigma$ entirely and keeps only the directions. The result has all singular values equal to 1: a (semi-)orthogonal matrix, an isometry on its row/column space, spectral norm exactly 1. The original paper calls this the "isomorphic" property — used loosely to mean *uniform across directions* (iso-morph = equal-form), not in the strict algebraic sense of structure-preserving bijection.

Mechanically, you never compute the SVD. Muon runs five iterations of a tuned quintic Newton-Schulz polynomial on the momentum matrix, which approximately pushes all singular values toward 1. The whole thing is matmuls — GPU-friendly, no expensive numerical linear algebra — and it works because the polynomial $p(x) = ax + bx^3 + cx^5$ with the right coefficients has a basin around $x \approx 1$ that's attractive for any input in $[0, \sigma_{\max}]$.

The reason this is the right thing to do, in the variational-norm framing: $UV^\top$ is exactly the maximizer of $\langle -M_t, \Delta\rangle$ over the spectral-norm unit ball. So Muon is steepest descent under a spectral-norm constraint, by construction.

### Why equalizing singular values is the right move (not the wrong one)

The obvious objection: shouldn't directions with larger gradient signal get larger updates? Isn't squashing all singular values to 1 throwing away useful information about which way to learn fastest?

The objection turns on what the singular values of $M_t$ actually mean. They don't represent "the most important features"; they represent the conditioning of the gradient flow itself. A direction can dominate the momentum spectrum because the parameter happens to sit at a large scale, because the loss surface is steep along it but the variable is already nearly optimal, because earlier layers' bad conditioning is leaking through, or simply because the gradient has been correlated along that direction for many steps and compounding has amplified it. None of those is "this is the right thing to learn faster."

This is the same observation that motivates Adam vs. SGD. Adam normalizes per coordinate by the running variance because raw gradient magnitudes are not a reliable signal of update importance. Muon normalizes per singular direction by the singular value for the same reason, but at the matrix level rather than the scalar level. Both are saying: trust the direction more than the magnitude, because the magnitude is contaminated by stuff that has nothing to do with how fast you should be learning that direction.

A useful framing: orthogonalization is *whitening* applied in the singular-vector basis of the momentum. Whitening doesn't claim that all features are equally important; it removes the spurious anisotropy introduced by the noise and correlation structure of the signal so that downstream learning sees a clean step. The Moonlight paper's SVD-entropy ablations confirm this empirically: trained networks under Muon end up with *more diverse* singular spectra in their weights, not less. The orthogonalization step prevents the optimizer from collapsing onto a few singular directions per step — over many steps, the network still allocates capacity to genuinely useful directions, but it does so through accumulated gradient signal in the right basis, not through a single dominant per-step direction.

### Consistent Update RMS, and where the spectral-norm framing softens

There's a complication. The orthogonalized update $UV^\top$ has spectral norm exactly 1, but its per-entry magnitude (its RMS) is shape-dependent. For a full-rank update on an $A \times B$ matrix,

$$\text{RMS}(UV^\top) \;=\; \frac{\|UV^\top\|_F}{\sqrt{AB}} \;=\; \frac{1}{\sqrt{\max(A, B)}}.$$

This is Lemma 1 of the MuonScalable / Moonlight paper. The proof drops out of the singular-value structure: $\|UV^\top\|_F^2 = \min(A, B)$ (one per nonzero singular value), divide by $AB$ and take the square root.

The consequence at scale: a $4096 \times 4096$ MLP gets per-entry updates ~5.6× smaller than a $128 \times 128$ per-head projection under the same nominal step size. Big matrices under-train; small matrices over-train. Vanilla Muon scales badly to LLM-sized models for this reason, and the same problem shows up in encoder distillation as soon as you mix MHA, GQA, or MLA heads in the trunk.

The "Consistent Update RMS" fix is to multiply by $\sqrt{\max(A, B)}$ and a small global constant tuned to match AdamW's empirical update RMS:

$$\Delta W \;=\; \eta \cdot 0.2 \cdot O_t \cdot \sqrt{\max(A, B)}.$$

After this rescaling, every matrix in the network — square, rectangular, big, small — gets per-entry updates of RMS ≈ 0.2, which lands inside AdamW's empirical 0.2–0.4 range. The payoff is hyperparameter transfer: you can use the same `lr` and `weight_decay` you tuned for AdamW, and you can mix Muon (for 2D weights) and AdamW (for everything else) without retuning each group.

It's worth being honest about what this trades away. The pure spectral-norm-budget story says every layer's update should have $\|\Delta W\|_2 = \eta$ — one operator-scale step size network-wide. After the $\sqrt{\max(A,B)}$ rescaling, the spectral norm of the applied update is $0.2\eta\sqrt{\max(A,B)}$, which is layer-dependent. You can read this two ways: (i) as a *per-layer learning rate* $\tilde\eta_l \propto \sqrt{\max(A_l, B_l)}$ inside the spectral-norm framework, which is perfectly legal and doesn't change how direction is chosen; (ii) as a *concession from spectral to entry-wise budgeting* — Muon picks step directions with spectral geometry but sizes steps with entry-wise (AdamW-matching) geometry, because empirically networks seem to want roughly constant per-entry update magnitude across layers regardless of shape. Both readings are correct; they're the same equation viewed from different angles. The Moonlight paper picked entry-wise sizing because it makes hyperparameter sharing tractable, and for a distillation setup that's juggling teacher selection, loss weights, and head architecture, removing optimizer hyperparameter drift from the search space is worth a lot.

### Caveats: where the per-layer spectral view holds, where it doesn't

The "constrain each weight matrix's spectral change per step" story is rigorous only when the network is a composition of linear maps and pointwise Lipschitz-bounded nonlinearities. Each linear stage has a well-defined spectral norm; ReLU, GELU, tanh, sigmoid, LayerNorm, and RMSNorm are all bounded-Lipschitz; and Lipschitz constants compose multiplicatively, so bounding each $\|\Delta W_l\|_2$ bounds the network's global behavior to first order. There's no "spectral norm of the whole network" object to speak of — the network is nonlinear and has no SVD — but the Jacobian at any input does, and its spectral norm is bounded by the product of the per-layer norms times the activation Lipschitz constants. That's how a per-matrix step-size constraint translates to global stability.

Attention breaks this cleanly. The softmax over $QK^\top / \sqrt{d}$ involves a *quadratic* form in the activations, and the resulting map is not globally Lipschitz — its effective Lipschitz constant grows with activation scale (Kim et al. 2021 work this out formally for self-attention). Spectral-norm control on $W_Q$ and $W_K$ controls the projections themselves, but it does not bound the softmax-of-product downstream. This is why production transformers add QK-norm (RMSNorm on $Q$ and $K$ before the dot product), scaled initialization, and pre-LN on residual streams — those are stability mechanisms layered on top of the optimizer to handle the attention-shaped non-Lipschitz region. Muon doesn't claim to solve attention stability; it provides well-conditioned per-matrix updates, and the architecture handles the rest.

Conv2d is the easier case. A 2D convolution is fully linear; its weight tensor has shape $[C_{\text{out}}, C_{\text{in}}, k_H, k_W]$, but the operation is a linear map and has a well-defined SVD (computable exactly via FFT per Sedghi et al. 2019). The standard Muon convention is to flatten the kernel into the input dimension — $[C_{\text{out}}, C_{\text{in}} \cdot k_H \cdot k_W]$ — and orthogonalize that 2D matrix. This is a useful approximation rather than the exact operator-space SVD, but it captures the dominant amplification mode (channel mixing usually dwarfs spatial mixing in a standard kernel).

### The practical recipe

PyTorch's official `torch.optim.Muon` (added in 2.9) handles only 2D parameters and errors on anything else — there's no automatic conv2d flattening. The implementation includes the consistent-RMS scaling but behind an opt-in flag:

```python
muon_params = [
    p for n, p in model.named_parameters()
    if p.ndim == 2
    and "embed" not in n
    and "head" not in n
    and "norm" not in n
]
muon_ids = {id(p) for p in muon_params}
adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]

optimizer_muon = torch.optim.Muon(
    muon_params,
    lr=lr,
    weight_decay=weight_decay,
    adjust_lr_fn="match_rms_adamw",   # opt into the sqrt(max(A,B)) scaling
)
optimizer_adamw = torch.optim.AdamW(
    adamw_params, lr=lr, weight_decay=weight_decay,
)
```

The `match_rms_adamw` flag selects the $0.2\sqrt{\max(A,B)}$ scaling from the paper. The default (`"original"`) only does Keller Jordan's aspect-ratio correction $\sqrt{\max(1, A/B)}$ — fine for small models with mostly-square matrices, but it leaves you re-tuning learning rates at ViT-L and above. Always opt in for serious training.

Flash-attention varlen kernels are orthogonal to all of this — they're a forward-pass implementation and Muon never sees them; the projection matrices they read are 2D and get handled normally. If you have conv layers (a patch-embedding stem, a CNN-style vision tower, anything 4D), put them in the AdamW group; don't try to coerce a 4D tensor into Muon without thinking through the flattening and scaling explicitly.

For multi-teacher encoder distillation specifically, three points are worth highlighting. First, the per-teacher head MLPs are 2D — they go in the Muon group. The residual adapter design from earlier in this post is exactly the kind of architecture Muon handles cleanly: stacks of 2D linear projections (Muon group) interleaved with LayerNorms whose scalar gain/bias parameters go in the AdamW group. Second, the patch-embedding conv at the front of a ViT goes in AdamW; the cost is marginal because it's a single small layer. Third, the consistent-RMS scaling matters most when you change architecture variants — MHA vs. GQA vs. MLA produce attention projection matrices of different shapes, and `match_rms_adamw` lets you swap variants without re-tuning the learning rate.

The honest summary: Muon is not magic, and there is no published encoder-distillation benchmark to quote a precise number against. In the language-model regime where the optimizer *has* been benchmarked, it reaches a given training loss roughly 1.35× faster than AdamW (Keller Jordan's FineWeb speedrun) and similar order in the Moonlight ablations. Expect comparable but not identical gains for an encoder student — the more distinctively matrix-shaped your parameters, the better Muon does relative to AdamW. What it really buys you beyond wall-clock is *interpretability of the optimizer*: every step is the spectral-norm-steepest direction with a known, layer-shape-aware budget. When something goes wrong in distillation — head collapse, dense-probe regression, one teacher's signal disappearing — you can rule out optimizer pathology cleanly, instead of wondering whether AdamW's per-coordinate adaptive step sizes have implicitly absorbed one of the loss imbalances you were supposed to be controlling explicitly. That decoupling is the underrated reason to switch.

## Evaluation: how to tell if the student actually inherited capability

A good encoder eval suite has three properties: it probes the *trunk* (not the distillation heads), it covers the axes the teachers were chosen for, and it includes at least one head-to-head with each teacher rather than just absolute numbers.

I think of it as four layers.

### 1. Teacher-faithfulness probes

For each teacher $t$, fit a *fresh* linear or 2-layer MLP probe from the student's trunk features to the teacher's output, on held-out images. Report the cosine similarity (or feature-prediction R²) and compare it to a sanity baseline — typically the same probe fit on top of a randomly initialized backbone, and on top of the teacher itself (which should saturate). This isolates "did the trunk preserve the structure of teacher $t$" from "did the distillation head memorize a mapping." If trunk-probe scores are far below the distillation-head scores you saw at train time, you have head collapse.

### 2. Frozen-feature downstream evals — the standard battery

Freeze the student trunk. Run the canonical probes:

- **ImageNet-1k linear probe and kNN** — the long-running sanity check. kNN especially is unforgiving of feature-space distortion.
- **VTAB-1k / VTAB-2** — 19 diverse downstream tasks, low-shot. Catches representations that are over-tuned to ImageNet semantics.
- **Linear probe on iNaturalist, Places365, ObjectNet** — fine-grained, scene, and distribution-shift respectively.
- **Detection:** COCO with a frozen-backbone ViTDet or simple FPN head. Sensitive to spatial localization quality.
- **Dense prediction:** the eval that tells you whether the student is *actually* an encoder rather than a fancy pooled embedding — covered in detail next.

#### Dense probing, in detail

This is the eval most worth understanding mechanically, because it's where multi-teacher distillation usually wins or quietly fails, and because the word "dense" hides what's actually happening.

A ViT turns an $H \times W$ image into an $h \times w$ grid of patch tokens (with patch size 14, a 518×518 image becomes a 37×37 grid of tokens — about 1,369 of them). A *dense linear probe* on ADE20K does this:

1. Freeze the backbone.
2. Pass an image through and grab the per-patch token features $z_{i,j} \in \mathbb{R}^d$ for every patch $(i, j)$ in the grid.
3. Apply a single 1×1 convolution (equivalently, a shared linear layer applied to each token independently) that maps $\mathbb{R}^d \rightarrow \mathbb{R}^{150}$ — one logit per ADE20K class, *per patch*.
4. Bilinearly upsample those logits from the $h \times w$ grid back to $H \times W$ to produce a pixel-wise segmentation map.
5. Train only the 1×1 conv against the ground-truth segmentation, evaluate mIoU.

That is the entire decoder. No DPT, no UperNet, no FPN, no attention. The only learnable parameters are the per-class projection. Anything the predictor "knows" about object boundaries, stuff/things distinctions, or spatial layout has to already live inside the backbone's per-patch features.

That is what "dense" means here: every patch token must independently carry enough signal to be classified into the right semantic class. Contrast this with ImageNet linear probing, which uses *one* feature vector per image (the CLS token or pooled output) and asks for *one* label. ImageNet linear probe rewards a backbone that produces a clean global summary; ADE20K dense linear probe rewards a backbone whose 1,369 individual patch features are each a clean local summary.

ADE20K is the canonical version of this test for three reasons. It has 150 classes covering both *things* (chair, person, lamp) and *stuff* (sky, road, sand), so the probe can't shortcut by relying on object-centric cues. Scenes are cluttered and multi-object, so the per-patch features have to disambiguate locally rather than leaning on a single dominant subject. And the resolution is high enough (typically evaluated at 512+) that patch-grid quality at the boundaries of small objects actually shows up in the metric. Cityscapes and Pascal VOC are the standard companions; depth on NYUv2 / KITTI is the same idea but with a continuous target and a different failure mode (smoothness vs. sharpness at depth edges).

#### SigLIP 2 vs. DINOv3 under dense probing

These two families illustrate the gap perfectly, and the reason it matters for distillation choices.

**SigLIP 2** is image-text contrastive (sigmoid loss, multilingual data). The training signal is global: a single image embedding is pulled toward its caption embedding. Patch tokens never receive a *local* learning signal — they only matter through the pooled embedding the contrastive loss sees. Empirically, SigLIP 2 is excellent on ImageNet linear probe, retrieval, and zero-shot classification, but its per-patch features carry mostly *image-level* semantics smeared across the grid. Run a dense linear probe on ADE20K and the features hold up only modestly; the gap to a self-supervised dense model is large, and it does not close at higher resolution.

**DINOv3** is the opposite design choice. Self-distillation with a Gram-matrix regularizer that explicitly penalizes patch-feature collapse, trained at scale on curated images. Every patch token is forced into a representation that is locally consistent and discriminative against other patches. The published DINOv3 ViT-L hits about **55.9 mIoU on ADE20K with just a linear dense probe** — a number that would have required a full UperNet decoder a couple of years ago, and that comes within striking distance of fully fine-tuned state-of-the-art (~63 mIoU). It also outperforms PEspatial by ~6 points and AM-RADIOv2.5 by ~3 on the same probe.

For SigLIP 2 at comparable scale, the linear ADE20K number lands meaningfully lower — the DINOv3 paper and follow-ups put weakly-supervised contrastive models (SigLIP 2, PEcore) clearly behind on this benchmark, and the gap *grows* at higher input resolution because contrastive backbones produce patch features that don't sharpen with more tokens, while DINO-style features do.

The takeaway for multi-teacher distillation is direct: if you only distill from SigLIP 2, no amount of clever loss design will produce strong ADE20K linear-probe numbers, because the teacher itself doesn't have that signal to give. The whole point of pairing a contrastive teacher with a DINO-family teacher is that the latter supplies dense locality and the former supplies language-alignable semantics. Your dense probe on the resulting student is the cleanest test of whether that pairing actually transferred — if your AM-RADIO–style student lands close to DINOv3's ADE20K number, the dense pathway worked; if it collapses toward SigLIP 2's number, your distillation heads ate the locality signal and you have head collapse on the dense side.

### 3. Capability-matched evals — one per teacher

Generic benchmarks miss the point of distilling a *specific* teacher. Add one targeted eval per teacher:

- **CLIP / SigLIP teachers** → zero-shot retrieval and classification through the student's CLIP-aligned head on COCO/Flickr30k retrieval and on ImageNet zero-shot. Compare to the original teacher at matched compute.
- **DINOv2 teacher** → unsupervised dense correspondence on SPair-71k or PF-Pascal, and overclustering / k-means purity on ImageNet — both stress the cleanliness of patch features.
- **SAM teacher** → mask quality with a fixed mask decoder driven by the student's encoder, evaluated on SA-1B held-out and COCO panoptic.
- **MAE teacher** → hold-one-out reconstruction and linear-probe on low-data regimes (1%, 10% ImageNet).

This is also where you check the *amalgamation* hypothesis: a student that genuinely combined teachers should match each teacher on its home turf. If the CLIP eval drops sharply, you didn't really distill CLIP — you distilled "things that look CLIP-like in the head."

### 4. Robustness, calibration, and efficiency

The final layer is the one papers under-report. Run the student through ImageNet-A, ImageNet-R, ImageNet-Sketch, and ObjectNet to check shift robustness. Measure expected calibration error after the linear probe. And quote real wall-clock numbers: throughput at the deployed resolution, parameter count, and FLOPs at the resolutions you actually serve, not at training resolution. A 10% gain that costs 3× compute is not a win.

## Reading the numbers honestly

A few habits that prevent self-deception:

Always report teacher numbers at the *student's compute budget*. Comparing a ViT-B student against a ViT-L teacher and declaring victory because the student is "close" elides the fact that you should be comparing to a from-scratch ViT-B trained the same way. Run that baseline.

Look at *gaps*, not levels. Distillation lifts every metric a bit; what matters is whether the student is closer to its teacher on dense tasks than a vanilla ViT-B is to that teacher.

Track **dispersion across seeds and probes**. Encoder evals have surprisingly high variance — linear probes on small VTAB tasks can move 1–2 points across seeds. A 0.5-point improvement on one benchmark is noise.

Do at least one **adversarial eval the student wasn't selected for** — e.g., depth estimation if you only optimized for segmentation. This is the closest thing we have to a held-out test of generality.

## Closing

Multi-teacher encoder distillation is one of the few places in deep learning where architectural choices, loss design, and evaluation methodology are tightly coupled: a clever loss looks like a win until you probe the trunk and discover the heads ate the gain. The discipline is to pair each teacher you add with a capability-matched eval that measures whether the *trunk* — not the distillation pipeline — actually inherited the thing you wanted. Do that, and the resulting student is genuinely a single backbone that punches above its weight; skip it, and you've trained a very expensive set of projection heads.
