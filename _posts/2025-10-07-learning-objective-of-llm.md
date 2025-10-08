---
title: Learning Objective of LLM
math: true
---

## Learning Objective of LLM

The fundamental of llm is to predict next token. In pretrain and SFT, given a sentence as the ground truth,
we want the output token at each step more likely to be the token in the ground truth. Let's 
denote the sentence as $\vec{\text{text}}$. Our objective is to maximize the $P(\vec{\text{text}})$.

Let's denote the sequence of token in the sentence up to the $(t-1)^{\mathrm{th}}$ token $s_t$.
Let's denote the $t^{\mathrm{th}}$ token $a_t$.

$$
P(\vec{\text{text}}) = \prod P(a_t \mid s_t)
$$

So our goal is to maximize this product. But these probability numbers are extremely small. Multiply them together
would be close to zero and subject to the hardware round off problem. Instead, we take the log of that probability.
There're two nice thing about log. It's monotonically increasing, meaning that maximize probability is the same as
maximize its log. log turns very small number to much managerable number. So we can write our objective function in
a more manageable way.

$$
\log P(\vec{\text{text}}) = \sum \log P(a_t \mid s_t)
$$

Notice that this expression is negative because probability is smaller than 1. We add a negative sign before that.
So maximize the probabilty is now minimize the negative log prob. So we get our loss function.

$$
\mathrm{Loss} = - \sum \log P(a_t \mid s_t)
$$

### Calculate the gradient for each logit
Let's set the following notation.

$$
\begin{aligned}
y_t &= \log P(a_t \mid s_t) \\
p_t &= P(a_t \mid s_t)
\end{aligned}
$$

Then we can apply chain rule.

$$
\begin{aligned}
\frac{\partial \mathrm{Loss}}{\partial y_t} &= -1 \\
\frac{\partial y_t}{\partial p_t} &= \frac{1}{p_t} \\
\frac{\partial \mathrm{Loss}}{\partial p_t} &= \frac{\partial \mathrm{Loss}}{\partial y_t} \frac{\partial y_t}{\partial p_t} = -\frac{1}{p_t}
\end{aligned}
$$

I will abuse my notation a bit. From now on, we use $p(k)$ to denote $P(a_t \mid s_t)$ where $a_t$ is the $k^{\mathrm{th}}$
token in the dictionary. $p(k)$ is the output of softmax function applied to the $k^{\mathrm{th}}$ logit, $z(k)$.

$$
\begin{aligned}
p(k) &= \frac{e^{z(k)}}{\sum_j e^{z(j)}} \\
\frac{\partial p(k)}{\partial z(i)}  &= \frac{e^{z(k)} \,\delta_{ik}\, \sum_j e^{z(j)} - e^{z(k)} e^{z(i)}}{\left[\sum_j e^{z(j)}\right]^2} \\
&= \frac{\big(\delta_{ik} \sum_j e^{z(j)} - e^{z(i)}\big) e^{z(k)}}{\left[\sum_j e^{z(j)}\right]^2} \\
&= p(k)\,\big(\delta_{ik} - p(i)\big)
\end{aligned}
$$

Use chain rule to calculate the derivative w.r.t $t^{\mathrm{th}}$ logits.

$$
\frac{\partial \mathrm{Loss}}{\partial z_t(i)}
= \frac{\partial \mathrm{Loss}} {\partial p_t(k)} \frac{\partial p_t(k)}{\partial z_t(i)}
= -\frac{1}{p_t(k)} \, p_t(k)\, \big(\delta_{ik} - p_t(i)\big)
= p_t(i) - \delta_{ik}
$$

### Update parameters
Let's update the lm_head. lm_head is usually an unbiased linear layer. The output of lm_head is the logit $z$. Let's denote its weight
$W$ and input $x$.

$$
\begin{aligned}
z_t &= W x_t \\
z_t(i) &= \sum_j W_{ij}\, x_{t j} \\
\frac{\partial z_t(i)}{\partial W_{ik}} &= x_{t k}
\end{aligned}
$$

So the derivative of Loss w.r.t $W$ is

$$
\begin{aligned}
\frac{\partial \mathrm{Loss}}{\partial W_{ik}}
&= \sum_t \frac{\partial \mathrm{Loss}}{\partial z_t(i)} \frac{\partial z_t(i)}{\partial W_{ik}} \\
&= \sum_t \big(p_t(i) - \delta_{ik}\big) x_{t k}
\end{aligned}
$$
