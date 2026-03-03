# Why Router-Replay Is a Critical Fix for Training MoE Language Models with RL

*A look at why the routing mechanism in Mixture-of-Experts models breaks standard reinforcement learning — and how a deceptively simple idea called Router-Replay (R3) fixes it.*

---

Reinforcement learning has become the secret sauce behind some of the most capable AI models we've seen recently. From OpenAI's o-series to DeepSeek-R1, RL fine-tuning is how raw language models are turned into reasoning powerhouses. But there's a quiet, underappreciated problem lurking when you try to apply RL to the fastest and most efficient class of modern models: **Mixture-of-Experts (MoE) architectures**.

A paper published in October 2025 by Wenhan Ma and colleagues — *"Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers"* — puts a name to this problem and offers a clean, elegant fix. The fix is called **Rollout Routing Replay**, or **R3**. And once you understand why it's necessary, you'll wonder how anyone was training MoE models with RL without it.

---

## A Quick Primer: What Makes MoE Models Special (and Tricky)

Most large language models are "dense" — every parameter participates in every computation. Mixture-of-Experts models take a different approach: they have many specialized sub-networks (called "experts"), but only a small subset of them activate for any given token. A lightweight **router** network decides, for each token, which experts should handle it.

This design is brilliant from an efficiency standpoint. Models like DeepSeek V3, Qwen3-MoE, and Mixtral can have hundreds of billions of parameters while only activating a fraction of them per token, making inference much cheaper than equivalently-capable dense models.

But this architectural cleverness introduces a subtle and dangerous complication when you try to apply reinforcement learning.

---

## The Problem: Two Different Routers, Two Different Worlds

When you train a language model with RL — say, using GRPO or PPO — the process looks roughly like this:

1. **Rollout phase**: The model generates responses to prompts. This runs on an *inference engine* (like vLLM or SGLang), optimized for fast text generation.
2. **Training phase**: The model computes log-probabilities for those same generated responses, and the RL algorithm uses them to update the model's weights.

In a dense model, these two phases are essentially equivalent. If you feed the same tokens through the model twice, you get the same probabilities. The math works out cleanly.

In an MoE model, something goes wrong. When you feed the same token through the model during training, **the router doesn't necessarily pick the same experts it picked during inference**. And if different experts are activated, you get different hidden states, different output logits, and crucially — **different token probabilities**.

This discrepancy has two sources:

**Nondeterminism in Top-K routing.** Most MoE routers select the top-K experts by score for each token. When two experts have very similar scores, which one "wins" can depend on floating-point precision, GPU parallelism implementation details, and the specific order of operations. The training engine and inference engine are typically separate systems with different implementations — so even for identical inputs, they can make different routing decisions.

**Architectural separation.** Inference engines are heavily optimized for throughput and latency. They often use different fused kernels, different precision modes, and different batching strategies than the training engine. These differences compound the routing divergence.

The result is that the probabilities the training engine computes for tokens are *different* from the probabilities that actually generated those tokens during rollout. And this mismatch is not small — the paper shows that it can cause dramatically divergent token probability ratios, with many tokens exhibiting extreme probability discrepancies between the two phases.

We can quantify how much of this KL divergence comes from routing specifically by comparing dense and MoE models on the same FSDP2 + vLLM stack:

| Model | Baseline KL | With R3 | Interpretation |
|---|---|---|---|
| LFM 1.2B Dense | ~0.003 | — | Framework noise floor |
| LFM MoE 8B | ~0.01–0.02 | ~0.004–0.007 | R3 brings MoE down toward dense floor |
| Qwen 30B MoE | ~0.003–0.004 | ~0.001 | Baseline already near dense floor |

The dense model establishes a **framework noise floor** (~0.003 nats) — irreducible KL from numerical differences between FSDP2 and vLLM unrelated to routing. LFM MoE 8B sits 3–6x above this floor, confirming the excess KL comes from routing disagreements. Interestingly, Qwen 30B MoE's baseline KL is already near the dense floor, suggesting its routing is more deterministic across frameworks — R3's benefit is model-dependent.

---

## Why This Breaks RL Training

RL algorithms like PPO and GRPO use something called an **importance sampling ratio** — essentially the ratio of the model's *current* probability for a token to the *old* probability from when the rollout was generated. This ratio tells the algorithm how much to trust each training example. If the ratio is close to 1, the distribution hasn't changed much and the gradient update is reliable. If the ratio is extreme — say, 10x or 0.1x — something has gone very wrong.

In practice, RL algorithms clip this ratio (typically in the range [0.8, 1.2] or similar) to prevent catastrophically large gradient updates. This is the "trust region" concept at the heart of PPO.

Here's the problem: in MoE models, the routing mismatch can push these importance sampling ratios to extreme values *before any learning even happens*, just from the discrepancy between the inference and training routers. From the algorithm's perspective, it looks like the model has already changed dramatically from the rollout policy — so either the gradients get clipped into meaninglessness, or (if clipping isn't aggressive enough) the training is hit with enormous, destabilizing updates.

This effect is especially dangerous with **multi mini-step training** — where you take multiple gradient steps per rollout batch (`ppo_mini_batch_size < train_batch_size`). Each mini-step shifts the policy further from the rollout distribution, and routing errors compound on top of that drift. We trained an LFM MoE 8B model on multi-turn GSM8K with 4 mini-steps per rollout, and the KL divergence between rollout and training tells the story clearly:

![Rollout KL comparison — baseline vs R3](/img/rollout_corr_kl_lfm_moe_baseline_vs_r3_comparison.png)
*Baseline KL fluctuates at 0.008–0.015 with spikes to 0.02; R3 holds steady at 0.003–0.005.*

In severe cases, this leads to **catastrophic training collapse**. Running the same configuration to 3,000 steps, the baseline enters a feedback loop — routing errors produce incorrect gradients, which destabilize weights, which cause even worse routing, which produces even larger errors — until the model irreversibly degenerates:

![3k steps grad norm — baseline explodes](/img/lfm_moe_3k_steps_grad_norm_compr.png)
*Baseline gradient norms begin spiking around step 2k and explode to 4000+ by step 2.5k. R3 remains flat near zero throughout.*

![3k steps KL — baseline breakdown](/img/lfm_moe_3k_steps_rollout_corr_kl_compr.png)
*Baseline `rollout_corr/kl` explodes to 30+ nats — a complete breakdown in logprob agreement between rollout and training. R3 stays near zero.*

![3k steps validation — baseline drops to zero](/img/lfm_moe_3k_steps_validation_compr.png)
*Baseline validation accuracy drops from ~0.89 to 0.0 after step 2.5k. R3 holds steady at ~0.90.*

![3k steps score mean — baseline collapses](/img/lfm_moe_3k_steps_score_mean_compr.png)
*The model can no longer produce correct answers. R3 maintains ~1.0 score mean.*

**The key trigger is multi mini-step training**, not multi-turn prompting. Single mini-step runs on the same model and task converge fine without R3 — the importance ratio stays near 1.0 regardless of routing noise, so the small KL perturbation from routing never reaches the clipping boundary.

---

## Previous Workarounds and Their Limitations

Before R3, researchers tried a few approaches to manage this problem:

**Group Sequence Policy Optimization (GSPO)** applies importance sampling corrections at the sequence level rather than the token level, smoothing out some of the noise. It helps, but doesn't address the root cause.

**Truncated Importance Sampling (TIS)** aggressively clips extreme probability ratios. This stabilizes training but at the cost of discarding potentially useful gradient signal — essentially papering over the symptom.

**Reducing nondeterminism at the kernel level** is another approach: some teams have introduced specialized GPU compute kernels designed to make routing more deterministic. This works, but it requires deep infrastructure changes and can impose performance overhead.

What none of these approaches do is fix the fundamental issue: the training engine is working with different routing decisions than the inference engine made. They're all workarounds that manage the consequences, rather than eliminating the cause.

---

## R3: The Elegant Fix

Rollout Routing Replay (R3) takes a different approach that's almost obvious in hindsight: **just record what the router decided during inference, and replay those same decisions during training**.

Concretely, during the rollout phase, for every token in every generated sequence, R3 logs which experts the router selected. These routing masks — binary indicators of which experts were activated — are stored alongside the generated tokens and rewards.

When the training phase begins, instead of letting the training engine's router make fresh decisions, R3 feeds those recorded routing masks back in directly. The training engine uses the *same expert activations* that the inference engine used. This means the log-probabilities computed during training reflect the same computational path that actually produced the tokens — so the importance sampling ratios are accurate, the gradients are meaningful, and training is stable.

The beauty of R3 is that it's minimal. It doesn't change the model architecture. It doesn't require new RL algorithms. It doesn't impose any restrictive constraints on the training objective. It just adds a thin logging layer during rollout and a replay mechanism during the training forward pass. And it adds minimal overhead — the routing masks are small relative to the token data, and replaying them adds negligible latency to the training forward pass.

### How Much to Replay: Indices vs. Indices + Weights

There's a practical design question in implementing R3: should you replay just the **routing indices** (which experts were selected) or also the **combination weights** (the softmax scores used to mix expert outputs)?

The original R3 paper describes replaying the full routing distribution. But VeRL's standard implementation only replays indices, letting the training engine recompute the combination weights. We tested both variants on LFM MoE 8B:

![Indices vs indices+weights KL comparison](/img/with_rollout_weight_r3_vs_baseline.png)

| Variant | rollout_corr/kl |
|---|---|
| Baseline (no replay) | ~0.01–0.02 |
| Indices only | ~0.006–0.010 |
| Indices + weights | ~0.004–0.007 |

Replaying weights provides additional KL reduction — it eliminates one more source of numerical disagreement between the two forward passes. However, there's a **staleness tradeoff**: indices-only lets the updated model recompute its own combination weights, preserving its current judgment about how to weight experts. Indices + weights forces the model to use stale interpolation coefficients from the rollout policy. The storage overhead is also higher (float tensors vs. integer tensors).

For most practical purposes, **indices-only replay captures the bulk of R3's benefit** while keeping the implementation simple.

---

## Why This Matters Beyond the Paper

If R3 were just an academic result, it would still be interesting. But it's clearly already making its way into production systems.

DeepSeek V3.2 — one of the most capable open-weight models available — explicitly adopted a routing replay technique in its RL training pipeline. Their technical report describes logging which experts were activated during rollout and forcing the same routing pattern during training, so gradient updates are attributed to the experts that actually produced the sampled answers. This is R3 in practice.

This adoption by a leading AI lab is a strong signal that router-replay isn't just theoretically useful — it's a necessary ingredient for reliably training MoE models with RL at scale.

As the field continues to push toward larger MoE models (the efficiency advantages are too good to ignore) and more sophisticated RL training (the capability gains are too good to ignore), the problem R3 solves will only become more acute. Any team training a frontier MoE model with RL is going to encounter this issue eventually.

---

## The Bigger Picture

There's something worth pausing on here. The fact that MoE models create this training-inference discrepancy wasn't obvious before people started running RL at scale on them. It's the kind of problem that only surfaces when you combine two relatively new techniques — sparse expert routing and RL fine-tuning — in ways that weren't anticipated when either was designed.

R3 is a reminder that as AI systems become more complex and composed from more intricate pieces, we should expect more of these subtle interaction effects. The solution, often, isn't to abandon either technique — it's to carefully characterize the interaction and build a targeted fix.

In this case, the fix is just a few lines of logging and replay logic. But without it, some of the most capable AI training pipelines in the world would be quietly, invisibly broken.

---

*The original paper — "Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers" by Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, and Fuli Luo — is available on arXiv at [2510.11370](https://arxiv.org/abs/2510.11370).*
