# Async Rollout in Verl

In previous posts we discussed how RL training for LLMs works: sample responses from the policy, score them with a reward model, and update the policy to favor higher-reward outputs. The computational pattern is straightforward—generate a batch, compute advantages, run a few gradient steps—but the devil is in the timing. Generation and training compete for the same GPUs, and one slow sample can stall the entire batch.

Verl is an open-source RL training framework (the public implementation of the HybridFlow paper, accepted at EuroSys 2025) that tackles this problem through async rollout. This post examines how Verl decouples generation from training and the algorithmic corrections needed to make off-policy updates work.

## The Long-Tail Problem

Synchronous RL waits for every sample in a batch to finish generating before training begins. The problem is that generation lengths follow a heavy-tailed distribution: most responses are short, but a few drag on. While those stragglers finish, the GPUs allocated to the other samples sit idle.

The issue compounds with longer contexts and agentic tasks. A model interacting with a code sandbox or web browser might wait seconds for environment feedback, and those waits are unpredictable. Simply adding more GPUs does not help—you cannot parallelize away the tail of a single slow trajectory.

## Colocated vs. Disaggregated

Verl supports two deployment modes that make different trade-offs between latency and utilization.

**Colocated (HybridEngine).** Training and generation share the same GPU pool. When generation runs, the optimizer states offload to CPU; when training runs, the inference engine pauses. Weight synchronization is fast—just an in-memory NCCL transfer—but the two workloads cannot overlap. This mode works well when generation is predictable and roughly matches training time.

**Disaggregated.** Generation and training run on separate node pools connected by queues. The generator streams samples to the trainer as they complete, so slow trajectories no longer block fast ones. The cost is network transfer for weights and samples, but the benefit is that both pools stay busy. Disaggregation also enables elastic scaling: you can add generator nodes during high-variance workloads without touching the trainer.

## Staleness and Off-Policy Drift

Once generation and training decouple, they inevitably drift apart. The generator produces samples under policy version $k$ while the trainer has already updated to version $k+1$ or beyond. PPO assumes on-policy data, so using stale samples without correction biases the gradient.

Verl controls this drift with a **staleness threshold** $\eta$. Setting $\eta = 0$ recovers synchronous training: the trainer blocks until fresh samples arrive. Setting $\eta > 0$ allows a fraction of the batch to come from older policy versions. The generator produces at most

$$
N_{\text{rollout}} = (1 + \eta) \cdot B - N_{\text{stale}}
$$

samples between weight syncs, where $B$ is the batch size and $N_{\text{stale}}$ counts leftover stale samples from the previous round. When generation is fast relative to training, $\eta = 1$ approximates one-step off-policy updates.

## Importance Sampling Corrections

Stale samples require importance sampling (IS) to reweight their contribution. Let $\pi_{\text{rollout}}$ denote the policy that generated a sample and $\pi_{\text{old}}$ the policy at the start of the current training step. The per-token IS ratio is

$$
\rho_t = \frac{\pi_{\text{old}}(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}.
$$

If $\pi_{\text{rollout}} = \pi_{\text{old}}$, the ratio is 1 and we recover standard PPO. When they differ, we have two correction strategies.

**Token-TIS (Token-Level Truncated IS).** Apply a clipped ratio at each token:

$$
w_t = \min(\rho_t, C_{\text{IS}}).
$$

The REINFORCE gradient becomes

$$
\nabla_\theta \mathcal{L}_{\text{Token-TIS}} = -\mathbb{E}_t \bigl[ \text{sg}(w_t) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \bigr],
$$

where $\text{sg}$ is stop-gradient. Token-TIS has low variance because each token gets its own weight, but it introduces $O(T^2 \Delta_{\max})$ bias when the policy drifts significantly (here $T$ is sequence length and $\Delta_{\max}$ is the maximum per-token divergence).

**Seq-TIS (Sequence-Level Truncated IS).** Compute a single weight for the entire trajectory:

$$
w_{\text{seq}} = \min\Bigl( \prod_t \rho_t,\; C_{\text{IS}} \Bigr) = \min\Bigl( \exp\bigl(\textstyle\sum_t \log \rho_t\bigr),\; C_{\text{IS}} \Bigr).
$$

This weight broadcasts to all tokens in the sequence. Seq-TIS is unbiased—it correctly reweights the full trajectory—but has higher variance because a single large ratio can dominate.

The choice depends on drift severity:

| Drift Level | Recommended Correction |
|-------------|------------------------|
| Negligible (same checkpoint) | None (bypass mode) |
| Moderate (slight staleness) | Token-TIS |
| Severe (replay buffer, old data) | Seq-TIS or Seq-MIS |

Seq-MIS is a variant that rejects (masks) sequences exceeding the IS threshold rather than clipping. It acts as a hard trust-region filter when the distribution tail contains garbage samples.

## Fully Async Training

Verl's fully async mode pushes decoupling further: the generator and trainer run continuously without explicit synchronization points. Samples flow through a queue, and the trainer consumes whatever is available. Weight updates push to the generator whenever a training step completes.

The key insight is that `old_log_prob`—the log-probability used for importance sampling—must be computed by the generator at rollout time, not recomputed by the trainer. This ensures the IS ratio correctly reflects the policy that actually produced the sample, even if the trainer has moved on.

In experiments on Qwen2.5-7B with 128 GPUs, fully async training achieved 2.35–2.67× throughput over synchronous baselines. On Qwen3-30B-A3B, the disaggregated async setup delivered 1.7× improvement compared to colocated mode.

## Takeaways

- Synchronous RL suffers from the long-tail problem: slow samples idle the entire batch.
- Disaggregated deployment separates generation and training onto different node pools, enabling overlap and elastic scaling.
- Staleness thresholds control how much off-policy drift is tolerable.
- Token-TIS and Seq-TIS provide importance sampling corrections with different bias-variance trade-offs.
- Fully async training with proper IS corrections can more than double throughput on large-scale runs.

Sources:
- [Verl Documentation: Fully Async Policy Trainer](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
- [Verl Documentation: Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- [Verl Documentation: Mathematical Formulations](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
- [Verl GitHub Repository](https://github.com/volcengine/verl)
