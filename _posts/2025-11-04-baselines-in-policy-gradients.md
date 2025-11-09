# Baselines in Policy Gradients

Given that 
$$
\underset{a_t \sim \pi_\theta}{\mathbb{E}} [\nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t)] = 0 \\
\nabla J(\pi_{\theta}) = \underset{\tau \sim \pi_\theta}{\mathbb{E}}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t | s_t) \sum_{t'=t}^{T} R(s_{t'}, a_{t'})]
$$
Why is
$$
\nabla J(\pi_{\theta}) = \underset{\tau \sim \pi_\theta}{\mathbb{E}}[\sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t | s_t) (\sum_{t'=t}^{T} R(s_{t'}, a_{t'}, s_{t'+1})-b(s_t))]
$$
True?