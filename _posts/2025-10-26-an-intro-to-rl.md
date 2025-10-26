## An Intro to RL

In the previous post, [Rejection Sampling – Why It Matters in LLM Training]({% post_url 2025-10-19-rejection-sampling %}), we relied on a model to generate supervised fine-tuning (SFT) data. The problem, as we saw, is that rejection sampling throws most of the generated samples away. Every iteration produces only a handful of “winning” responses, which is painfully inefficient if you want to keep improving the model.

Reinforcement learning (RL) takes a different approach: instead of discarding the less-preferred samples, we keep all of them and weight their contribution by how good they are. Suppose we draw $S$ samples and score each one with a reward function $R(\vec{text}_i)$. The REINFORCE update multiplies the log probability of each sample by its reward:

$$
\text{Loss} = -\frac{1}{S} \sum_{i=1}^{S} R(\vec{text}_i) \log P(\vec{text}_i).
$$

Minimizing this loss is equivalent to maximizing the model’s expected reward,

$$
\text{REINFORCE Objective} = \mathbb{E}_{\vec{text} \sim P} [R(\vec{text})],
$$

which is a direct contrast to the SFT objective that implicitly assumes a binary reward:

$$
\text{SFT Objective} = \mathbb{E}_{\vec{text} \sim P} [\mathbf{1}_{A}(\vec{text})].
$$

### Why We Need a Baseline

The catch with the simple REINFORCE loss is that its gradient has very high variance—the update can swing wildly depending on the reward of a single sample. A classic fix is to subtract a baseline, a guess of how well the model would do on average in the same situation. We denote this baseline by $V_M(\vec{s}_t)$, the value of the current state (token prefix) under model $M$. After adding the baseline, the loss becomes

$$
\begin{aligned}
\text{Loss}
&= -\frac{1}{S} \sum_{i} \sum_t (R(\vec{text}_i) - V_M(\vec{s}_{it})) \cdot \log P(a_{it} | \vec{s}_{it}) \\
&= -\frac{1}{S} \sum_i \sum_t A_M(a_{it}, \vec{s}_{it}) \cdot \log \pi_M(a_{it} | \vec{s}_{it}),
\end{aligned}
$$

where we introduced two standard RL terms:

$$
A_M(a_t, \vec{s}_t) = R(\vec{text}) - V_M(\vec{s}_t), \qquad
\pi_M(a_t | \vec{s}_t) = P(a_t | \vec{s}_t).
$$

The subscript $M$ simply reminds us that these quantities come from our model. The advantage $A_M$ measures how much better an action is than the baseline; maximizing expected advantage

$$
\text{REINFORCE Objective} = \mathbb{E}_{\pi_M} [A_M(a_t, \vec{s}_t)]
$$

nudges the model toward actions that outperform its current average behavior.

### Value Function

We have been using the value function $V_M(\cdot)$ without defining it formally. Intuitively, it is the average reward the model expects if it continues generating tokens starting from the partial sequence $\vec{s}_{t}$. Formally, let $A_t$ denote a continuation $a_t, a_{t+1}, \ldots, a_T$ sampled from the model. Then

$$
\begin{aligned}
V_M(\vec{s}_t)
&= \mathbb{E}_{A_t \sim P(\cdot | \vec{s}_t)} [R(\vec{s}_t + A_t)] \\
&= \sum_{a_t} P(a_t | \vec{s}_t) \sum_{A_{t+1}} P(A_{t+1} | \vec{s}_{t+1}) R(\vec{s}_t + A_t) \\
&= \sum_{a_t} P(a_t | \vec{s}_t) V_M(\vec{s}_{t+1}).
\end{aligned}
$$

At the end of the sequence, $\vec{text} = \vec{s}_{T+1}$, so $V_M(\vec{s}_{T+1}) = R(\vec{text})$. Put together, we have the familiar recursive relationship:

$$
\begin{aligned}
V_M(\vec{s}_t) &= \sum_{a_t} P(a_t | \vec{s}_t) V_M(\vec{s}_{t+1}) \quad \text{for } t = 0, \ldots, T, \\
V_M(\vec{s}_{T+1}) &= R(\vec{text}).
\end{aligned}
$$

In theory we could compute the baseline exactly with this recursion, but in practice that is hopeless for large models. Instead we train a separate neural network to approximate $V_M$. This network is the **critic**, while the original model that proposes actions is the **actor**—hence actor-critic methods.

### Reward Model

To finish the loop we need a reward function. Recall the preference model from the rejection sampling post: it took two responses and predicted which one humans prefer. A reward model simplifies that setup. It looks at a single response and outputs a scalar reward that should be consistent with those same human judgments.

We can reuse the preference data by enforcing that the preferred response gets the higher reward. If we score responses $i$ and $j$ with rewards $R(\vec{text}_i)$ and $R(\vec{text}_j)$, we define the preference probability via the logistic link:

$$
\begin{aligned}
P(\vec{text}_i) &= \frac{\exp(R(\vec{text}_i))}{\exp(R(\vec{text}_i)) + \exp(R(\vec{text}_j))} \\
&= \frac{1}{1 + \exp(-(R(\vec{text}_i) - R(\vec{text}_j)))} \\
&= \sigma(R(\vec{text}_i) - R(\vec{text}_j)),
\end{aligned}
$$

where $\sigma$ is the sigmoid function. Training the reward model is then just binary cross entropy on these pairwise comparisons:

$$
\begin{aligned}
\text{Reward loss}
&= -hp \cdot \log P(\vec{text}_i) - (1 - hp) \cdot \log P(\vec{text}_j) \\
&= -\log P(\vec{text}_w) \\
&= -\log \sigma(R(\vec{text}_w) - R(\vec{text}_l)),
\end{aligned}
$$

where $hp = 1$ if humans preferred response $i$ and $hp = 0$ otherwise. Here $\vec{text}_w$ is the human-preferred response and $\vec{text}_l$ is the other response. Once this reward model is trained, it can score any new sample, letting RL reuse every generated token instead of throwing most of them away.
