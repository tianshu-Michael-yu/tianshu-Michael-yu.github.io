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
- *Registers are usually not distilled directly*. They're internal scratch space for the student trunk. What you want is for the student's registers to soak up the same garbage that the teacher's would have soaked up — which happens implicitly when you supervise the patch grid. Trying to align the student's registers to a teacher's registers is brittle (the index ordering is arbitrary) and tends to hurt.

A subtler point: AM-RADIO ablated using a *separate* student CLS token vs. average-pooling the student's patch tokens for the summary loss. Average-pooling looks tempting (one fewer token), and it improves the summary loss in isolation — but it *hurts* the feature (patch) loss, because you're now leaning on the patch grid to do double duty. They picked separate CLS tokens for exactly this reason. The general lesson: give each loss its own dedicated tokens, or you get cross-talk where one objective degrades another's signal.

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
