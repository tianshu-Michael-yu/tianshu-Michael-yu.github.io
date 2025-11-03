## RL evolution - TRPO and PPO
Based on our previous post, [An Intro to RL]({% post_url 2025-10-26-an-intro-to-rl %})
Let's outline the algorithm of REINFORCE.

1. Generate a dataset $S$ from the current model $M$.
2. Update the parameters of model $M$ a few times by minimizing the loss $-\frac{1}{S} \sum_i \sum_t A_{M}(a_{it}, s_{it}) \cdot \log \pi_{M}(a_{it} | s_{it})$.
3. Repeat step 1 and 2

The REINFORCE algorithm iteratively optimize $\mathbb{E}_{s,a \sim \pi_M}[A_M(a, s)]$.

### Trusted Region Policy Optimization (TRPO)

There are few problems with the above procedures. Let's say $M_0$ is the model at step 1 and $M_1$ the current model that's being updated in step 2. We are drawing samples from $M_0$ but the model we hope to improve is $M_1$. The distribution of the two models don't match. We can solve this problem by importance sampling, which is a technique that evaluting properties of a distribution $P$ through another distribution $Q$. 

Suppose we want to calculate of the expectation of a random variable X when it's sampled from a distribution given by $p(x)$. But we draw our sample from a distribution given by $q(x)$. We can do the following. 

$$
\mathbb{E}_{x \sim p(x)}[X] = \sum_x p(x) x = \sum_x q(x) \frac{p(x)}{q(x)} x = \mathbb{E}_{x \sim q(x)}[\frac{p(X)}{q(X)} X]
$$

Substitue that to the optimization objective of REINFORCE. 

$$
\mathbb{E}_{s,a \sim \pi_1}[A_1(a, s)] = \mathbb{E}_{s,a \sim \pi_0}[\frac{\pi_1(a|s)}{\pi_0(a|s)} A_0(a, s)]
$$

Here $\pi_1$ is a shorthand notation for $\pi_{M_1}$ and $A_1$ is a shortand notation for $A_{M_1}$.

With our new objective, we can update our loss.

$$
\text{Loss} = -\frac{1}{S} \sum_i \sum_t \frac{\pi_1(a_{it} | s_{it})}{\pi_0(a_{it} | s_{it})} A_0(a_{it}, s_{it})
$$

Notice that we didn't take the $\log$ over probability, because it's already a ratio.

Now consider a situtation where the reward model is trained on pairs of responses where it learns to detect certain featrues of responses that are preferrable to human and it has only ever been presented with high quality resposnes, having never seen bad quality content. As a consequence, the reward model focus on formatting to distinguish preferred responses. The reward model can give high reward to well-formatted low-quality incoherent responses, cuasing the main model to go into a destructive spiral of prioritizing incoherent well-formatted responses to maximize reward. This is typical illustration of reward hacking. What we can here is that we can restrict our optimization so that the new model did not go very far from the original model.

Let's formalize the idea. What we want is the behavior of our model doesn't deviate too far from our original model. That means the sample distribution of our updated model shouldn't suprise us too much if we pretend it to be sample from our original model. This is captured nicely in a measure of distance called KL divergence, commonly denoted by $D_{KL}(p||q)$ for two distributions $p$ and $q$. The definition is the following.

$$
\begin{aligned}
D_{KL}(p||q) &= \sum_x p(x)\log \frac{p(x)}{q(x)} \\
&= (-\sum_x p(x) \log q(x)) - (-\sum_x p(x) \log p(x))
\end{aligned}
$$

Look at the second summand and you realize that's the definition of the entropy of the distribution $p$. So the KL divergence is also a measure of relative entropy. Notice that the definition of KL divergence is not symmetric, meaing that there exist $p$ and $q$ such that $D_{KL}(p||q) \neq D_{KL}(p||q)$.


The KL divergence in our case turns out to be:

$$
D_{KL}(\pi_0(\cdot|s)|| \pi_1(\cdot|s)) = \sum_a \pi_0(a|s) \log(\frac{\pi_0(a|s)}{\pi_1(a|s)})
$$

We can simply add this term as a penalty to the loss function which will cause the optimization to penalize movements far away from the starting distribution. Since we don't know if the scale of our loss function matches that of the calculated KL penalty, we will need a scaling factor here. So the TRPO loss can look like this:

$$
\text{TRPO Loss} = -\frac{1}{S} \sum_i \sum_t \frac{\pi_1(a_{it} | s_{it})}{\pi_0(a_{it} | s_{it})} A_0(a_{it}, s_{it}) + \beta \cdot D_{KL}(\pi_0(\cdot|s_t)||\pi_1(\cdot|s_t))
$$

However, it's hard to balance the scaling factor $\beta$ to get step sizes that are not too small. One way to take larger step sizes more robustly is to use a constraint on KL divergence instead of putting it in the loss. This makes the TRPO optimization problem to be:

$$
\begin{aligned}
\text{minimize over }M_1 &: -\frac{1}{S} \sum_i \sum_t \frac{\pi_1(a_{it} | s_{it})}{\pi_0(a_{it} | s_{it})} A_0(a_{it}, s_{it}) \\
\text{subject to constraint} &: D_{KL}(\pi_0(\cdot|s_{it})||\pi_1(\cdot|s_{it})) \le \delta, \; \forall i,t, \; \text{for some } \delta
\end{aligned}
$$

So if our optimizer following the above loss suggested a parameter update that causes the constraint to be violated, how should we modify the update so that it satisfy the constraint? Intuitively, our KL limit defines a trust-region, where parameter updates have to land in that region. The natural gradient direction defines a line starting from our model's current position to the suggested update in parameter space. We can calculate where the boundary of the trust-region and our line intercepts. That interception point is the maximal step we can take while staying in the trust-region. Obiviously, this is a hard problem, since we know that computing where to land exactly on a non-linear boundary is hard. 

### Proximal Policy Optimization (PPO)

PPO also put some limits to prvent the probability of $a_t$ from drifting two far away. But it achives this in a simpiler way than TRPO. One insight here is that if we capped the loss function every time the two probabilites were further apart than a certain value ($\epsilon$), then there would be no gain that the optimization process would achieve from moving the probabilities further apart.  Since the probability are already written as a ratio, we can simply make sure that ratio is in the neighborhood of 1 by clipping the ratio in the loss function and taking max (or min inside the negative sign). This give us the PPO Loss function:

$$
\text{PPO Loss} = -\frac{1}{S} \sum_i \sum_t \min [\frac{\pi_1(a_{it} | s_{it})}{\pi_0(a_{it} | s_{it})} A_0(a_{it}, s_{it}), \text{CLIP}(\frac{\pi_1(a_{it} | s_{it})}{\pi_0(a_{it} | s_{it})}, 1-\epsilon, 1+\epsilon)A_0(a_{it}, s_{it})]
$$

The CLIP function simply clips the provided value at both ends and, as such, $\text{CLIP}(v, x, y) = min(max(v,1),y)$.