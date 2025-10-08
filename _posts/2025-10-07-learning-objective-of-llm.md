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
There're two nice things about log. It's monotonically increasing, meaning that maximizing probability is the same as
maximizing its log. log turns very small numbers to much more manageable numbers. So we can write our objective function in
a more manageable way.

$$
\log P(\vec{\text{text}}) = \sum \log P(a_t \mid s_t)
$$

Notice that this expression is negative because probability is smaller than 1. We add a negative sign before that.
So maximizing the probability is now minimizing the negative log prob. So we get our loss function.

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

$\delta_{ik}$ is the Kronecker delta: it equals 1 when $i=k$ and 0 otherwise. It can be viewed as a one-hot target distribution. In fact, the above formula can 
be generalized to other target distributions as well. Let's denote the target distribution $\tau$ and the distribution for the $t^{th}$ output is 
$\tau_t$.

Use chain rule to calculate the derivative w.r.t $t^{\mathrm{th}}$ logits.

$$
\frac{\partial \mathrm{Loss}}{\partial z_t(i)} 
= \sum_k \frac{\partial \mathrm{Loss}}{\partial p_t(k)} \frac{\partial p_t(k)}{\partial z_t(i)}
= \sum_k \left(-\frac{1}{p_t(k)}\right)p_t(k)(\tau_t(i) - p_t(i))
= p_t(i) - \tau_t(i)
$$

### Update parameters

Let's update the lm_head. lm_head is usually an unbiased linear layer. The output of lm_head is the logit $z$. Let's denote its weight
$W$ and input $x$.

$$
z_t = W x_t \\
z_t(i) = \sum_j W_{ij}\, x_{t j} \\
\frac{\partial z_t(i)}{\partial W_{ik}} = x_{tk} \\
\frac{\partial z_t(i)}{\partial x_{tk}} = W_{ik}
$$

So the derivative of Loss w.r.t $W$ is

$$
\begin{aligned}
\frac{\partial \mathrm{Loss}}{\partial W_{ik}}
&= \sum_t \frac{\partial \mathrm{Loss}}{\partial z_t(i)} \frac{\partial z_t(i)}{\partial W_{ik}} \\
&= \sum_t \big(p_t(i) - \tau_t(i)\big) x_{t k}
\end{aligned}
$$

Let's write out the updated parameter $W'$ using SGD.

$$
\begin{aligned}
W_{ik}' &= W_{ik}-\eta  \frac{\partial Loss}{\partial W_{ik}} \\
&= W_{ik} - \eta  \sum_t (p_t(i) - \tau_t(i))x_{tk}
\end{aligned}
$$

Let's also compute the derivative of Loss w.r.t $x$.

$$
\begin{aligned}
\frac{\partial Loss} {\partial x_{tk}} &= \sum_i \frac{\partial Loss}{\partial z_t(i)} \frac{\partial z_t(i)}{\partial x_{tk}} \\
&= \sum_i (p_t(i) - \tau_t(i)) W_{ik}
\end{aligned}
$$

### Exercise

As you may notice, in order to do the backprogation we need to store $p_t(i)$. But this is very large. A typical tokenizer's vocab size
is 150K. During pretrain for 30B model, we typically have around 1M tokens per batch. We can calculate the size of $P$.

$$
\begin{aligned}
Sizeof(P) &= vocab\_size \times num\_tokens \times Sizeof(fp32) \\
&= 150K \times 1M \times 4\ \text{bytes} \approx 558\ \text{GB}
\end{aligned}
$$

A typical H100 only have 80 GB. If this matrix ever materialized, we will certainly run out of device memory. I will left you as an exercise
to figure out how we can calculate the derivative without having to store this matrix.