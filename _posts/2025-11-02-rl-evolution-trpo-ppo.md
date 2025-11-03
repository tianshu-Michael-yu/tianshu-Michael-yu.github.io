## RL evolution - TRPO and PPO

In the previous post, [An Intro to RL]({% post_url 2025-10-26-an-intro-to-rl %}), we built the background for policy-gradient methods. Here we continue that story, starting from REINFORCE and then motivating two improvements: Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO).

To recap, the REINFORCE loop looks like this:

1. Collect a dataset $S$ of trajectories with the current policy $M$.
2. Update the parameters of $M$ for a few gradient steps by minimizing the loss $$-\frac{1}{|S|} \sum_i \sum_t A_M(a_{it}, s_{it}) \log \pi_M(a_{it} \mid s_{it})$$.
3. Repeat the process with fresh data.

REINFORCE performs stochastic gradient ascent on the objective $\mathbb{E}_{s,a \sim \pi_M}[A_M(a, s)]$, where $A_M$ is an estimate of the advantage under policy $M$.

### Trust Region Policy Optimization (TRPO)

This basic loop hides a subtle problem. Let $M_0$ denote the policy that generated the data in step 1, and let $M_1$ be the updated policy in step 2. Although we optimize $M_1$, the trajectories come from $M_0$, so the gradient is biased. The standard fix is importance sampling, a technique for estimating expectations under one distribution using samples from another.

Suppose we want $\mathbb{E}_{x \sim p}[f(x)]$ but only have samples from $q$. We can rewrite the expectation as:

$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_x p(x) f(x) = \sum_x q(x) \frac{p(x)}{q(x)} f(x) = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right].
$$

Applying this to REINFORCE with $\pi_0 = \pi_{M_0}$ and $\pi_1 = \pi_{M_1}$ gives:

$$
\mathbb{E}_{s,a \sim \pi_1}[A_1(a, s)] = \mathbb{E}_{s,a \sim \pi_0}\left[\frac{\pi_1(a \mid s)}{\pi_0(a \mid s)} A_0(a, s)\right].
$$

This leads to a weighted loss:

$$
\text{Loss} = -\frac{1}{|S|} \sum_i \sum_t \frac{\pi_1(a_{it} \mid s_{it})}{\pi_0(a_{it} \mid s_{it})} A_0(a_{it}, s_{it}).
$$

Now imagine a reward model that has only ever seen high-quality responses, so it learns to equate polished formatting with good answers. When the policy is optimized against that model, it may produce incoherent but well-formatted text that receives a high reward. This is a classic reward-hacking failure: the new policy diverges too far from the behavior it was trained on. We want to prevent such runaway updates.

A natural approach is to limit how much the new policy can deviate from the old one. The Kullback–Leibler (KL) divergence, denoted $D_{KL}(p \Vert q)$, measures how surprised we would be to see samples from $p$ while pretending they came from $q$:

$$
\begin{aligned}
D_{KL}(p \Vert q) &= \sum_x p(x) \log \frac{p(x)}{q(x)} \\
&= (-\sum_x p(x) \log q(x)) - (- \sum_x p(x) \log p(x)).
\end{aligned}
$$

The second term is the entropy of $p$, so the KL divergence captures the extra surprise relative to the entropy of $p$. KL is asymmetric: generally $D_{KL}(p \Vert q) \neq D_{KL}(q \Vert p)$.

For our policies, the state-conditioned KL divergence is:

$$
D_{KL}(\pi_0(\cdot \mid s) \Vert \pi_1(\cdot \mid s)) = \sum_a \pi_0(a \mid s) \log \frac{\pi_0(a \mid s)}{\pi_1(a \mid s)}.
$$

We can add this quantity as a penalty to discourage large updates:

$$
\text{TRPO Loss} = -\frac{1}{|S|} \sum_i \sum_t \frac{\pi_1(a_{it} \mid s_{it})}{\pi_0(a_{it} \mid s_{it})} A_0(a_{it}, s_{it}) + \beta \, D_{KL}(\pi_0(\cdot \mid s_t) \Vert \pi_1(\cdot \mid s_t)).
$$

Choosing the penalty coefficient $\beta$ is delicate. TRPO’s key idea is to avoid tuning $\beta$ by directly constraining the KL divergence:

$$
\begin{aligned}
\text{minimize over } M_1 &: -\frac{1}{|S|} \sum_i \sum_t \frac{\pi_1(a_{it} \mid s_{it})}{\pi_0(a_{it} \mid s_{it})} A_0(a_{it}, s_{it}) \\
\text{subject to constraint} &: D_{KL}(\pi_0(\cdot \mid s_{it}) \Vert \pi_1(\cdot \mid s_{it})) \le \delta, \quad \forall i,t, \; \text{for some } \delta
\end{aligned}
$$

The KL constraint defines a trust region—parameter updates must land inside it. In practice, TRPO approximates this constrained optimization using natural gradients and a line search to find the largest feasible step. Computing the exact boundary intersection is expensive, so the algorithm relies on approximations that work well empirically.

### Proximal Policy Optimization (PPO)

PPO seeks a similar effect—keeping policy updates close to the data-collecting policy—but with a simpler optimization problem. Let

$$
r_t = \frac{\pi_1(a_t \mid s_t)}{\pi_0(a_t \mid s_t)}.
$$

If $r_t$ becomes very large or very small, the update may destabilize. PPO clips the ratio so that once it leaves the interval $[1 - \epsilon, 1 + \epsilon]$, additional movement produces no further gain:

$$
\text{PPO Loss} = -\frac{1}{|S|} \sum_i \sum_t \min \Bigl[r_{it} \, A_0(a_{it}, s_{it}), \, \text{CLIP}(r_{it}, 1 - \epsilon, 1 + \epsilon) \, A_0(a_{it}, s_{it})\Bigr].
$$

The clipping function is $\text{CLIP}(v, x, y) = \min(\max(v, x), y)$. By taking the minimum inside the loss, PPO chooses the more conservative update between the unclipped and clipped terms, ensuring that the policy cannot exploit large deviations in either direction. This yields a first-order method that is much easier to implement than TRPO while retaining most of its stability.
