## Learning Objective of LLM

In this post, we will explore the fundamental learning objective of Large Language Models (LLMs). We will start by understanding the goal of predicting the next token in a sequence, then delve into the mathematical formulation of this objective. Following that, we will derive the gradients needed for training the model and discuss practical challenges related to memory consumption during backpropagation. By the end, you will have a clear understanding of how the loss function is constructed and optimized in LLMs, as well as insights into efficient gradient computation.

### Predicting the Next Token: The Core Objective

At the heart of LLM training lies the task of predicting the next token in a sequence. During both pretraining and supervised fine-tuning (SFT), given a ground truth sentence, the model aims to assign higher probabilities to the correct token at each step. 

Let's denote the sentence as a sequence of tokens $\vec{\text{text}}$. Our objective is to maximize the probability of this entire sequence, denoted as $P(\vec{\text{text}})$.

We define the sequence of tokens up to the $(t-1)^\mathrm{th}$ token as $s_t$, and the $t^\mathrm{th}$ token as $a_t$. The probability of the sentence can be factorized as the product of conditional probabilities:

$$
P(\vec{\text{text}}) = \prod_t P(a_t \mid s_t)
$$

Maximizing this product directly is numerically unstable since these probabilities are very small and multiplying many of them leads to values close to zero, which can cause issues with hardware precision. To address this, we take the logarithm of the probability, which transforms the product into a sum:

$$
\log P(\vec{\text{text}}) = \sum_t \log P(a_t \mid s_t)
$$

The logarithm is a monotonically increasing function, so maximizing the log probability is equivalent to maximizing the original probability. Moreover, it converts very small numbers into more manageable values.

Since probabilities are less than or equal to 1, the log probabilities are negative or zero. To frame this as a minimization problem, we define the loss function as the negative log probability:

$$
\mathrm{Loss} = - \sum_t \log P(a_t \mid s_t)
$$

This loss function is commonly known as the negative log-likelihood loss.

### Calculating the Gradient for Each Logit

To train the model using gradient-based optimization, we need to compute the derivatives of the loss with respect to the model's outputs (logits). Let's introduce some notation:

$$
\begin{aligned}
y_t &= \log P(a_t \mid s_t) \\
p_t &= P(a_t \mid s_t)
\end{aligned}
$$

Applying the chain rule, we find the derivatives of the loss with respect to $y_t$ and $p_t$:

$$
\begin{aligned}
\frac{\partial \mathrm{Loss}}{\partial y_t} &= -1 \\
\frac{\partial y_t}{\partial p_t} &= \frac{1}{p_t} \\
\frac{\partial \mathrm{Loss}}{\partial p_t} &= \frac{\partial \mathrm{Loss}}{\partial y_t} \frac{\partial y_t}{\partial p_t} = -\frac{1}{p_t}
\end{aligned}
$$

Next, we relate these probabilities to the logits produced by the model. We denote $p(k)$ as the probability assigned to the $k^\mathrm{th}$ token in the vocabulary at step $t$. This is the output of the Softmax function applied to the logits $z(k)$:

$$
\begin{aligned}
p(k) &= \frac{e^{z(k)}}{\sum_j e^{z(j)}} \\
\frac{\partial p(k)}{\partial z(i)} &= \frac{e^{z(k)} \,\delta_{ik}\, \sum_j e^{z(j)} - e^{z(k)} e^{z(i)}}{\left[\sum_j e^{z(j)}\right]^2} \\
&= \frac{\big(\delta_{ik} \sum_j e^{z(j)} - e^{z(i)}\big) e^{z(k)}}{\left[\sum_j e^{z(j)}\right]^2} \\
&= p(k)\,\big(\delta_{ik} - p(i)\big)
\end{aligned}
$$

Here, $\delta_{ik}$ is the Kronecker delta, which equals 1 if $i = k$ and 0 otherwise. This can be interpreted as a one-hot target distribution. In fact, this formula generalizes to any target distribution $\tau$, where the distribution at step $t$ is $\tau_t$.

Using the chain rule, we calculate the derivative of the loss with respect to the logits at step $t$:

$$
\frac{\partial \mathrm{Loss}}{\partial z_t(i)} 
= \sum_k \frac{\partial \mathrm{Loss}}{\partial p_t(k)} \frac{\partial p_t(k)}{\partial z_t(i)}
= \sum_k \left(-\frac{1}{p_t(k)}\right) p_t(k) (\tau_t(i) - p_t(i))
= p_t(i) - \tau_t(i)
$$

This result shows that the gradient with respect to the logits is simply the difference between the predicted probability and the target distribution.

### Updating Parameters

Let's consider the update step for the parameters of the final linear layer, often called the lm_head, which produces the logits. Suppose the lm_head has weight matrix $W$ and input vector $x$ at step $t$:

$$
z_t = W x_t \\
z_t(i) = \sum_j W_{ij} x_{t j}
$$

The partial derivatives of the logits with respect to the weights and inputs are:

$$
\frac{\partial z_t(i)}{\partial W_{ik}} = x_{t k} \\
\frac{\partial z_t(i)}{\partial x_{t k}} = W_{ik}
$$

Using these, the gradient of the loss with respect to the weights is:

$$
\begin{aligned}
\frac{\partial \mathrm{Loss}}{\partial W_{ik}}
&= \sum_t \frac{\partial \mathrm{Loss}}{\partial z_t(i)} \frac{\partial z_t(i)}{\partial W_{ik}} \\
&= \sum_t \big(p_t(i) - \tau_t(i)\big) x_{t k}
\end{aligned}
$$

The weights are updated using stochastic gradient descent (SGD) with learning rate $\eta$:

$$
\begin{aligned}
W_{ik}' &= W_{ik} - \eta \frac{\partial \mathrm{Loss}}{\partial W_{ik}} \\
&= W_{ik} - \eta \sum_t (p_t(i) - \tau_t(i)) x_{t k}
\end{aligned}
$$

Similarly, the gradient of the loss with respect to the input $x$ is:

$$
\begin{aligned}
\frac{\partial \mathrm{Loss}}{\partial x_{t k}} &= \sum_i \frac{\partial \mathrm{Loss}}{\partial z_t(i)} \frac{\partial z_t(i)}{\partial x_{t k}} \\
&= \sum_i (p_t(i) - \tau_t(i)) W_{ik}
\end{aligned}
$$

### Practical Challenge: Memory Consumption During Backpropagation

In practice, to perform backpropagation, we need to store the predicted probability distributions $p_t(i)$ for all tokens at every step. However, this can be prohibitively large. For example, consider a tokenizer with a vocabulary size of 150,000 and a batch containing around 1,000,000 tokens during pretraining of a 30B parameter model. The size of the probability matrix $P$ is:

$$
\begin{aligned}
\mathrm{Sizeof}(P) &= \mathrm{vocab\_size} \times \mathrm{num\_tokens} \times \mathrm{Sizeof(fp32)} \\
&= 150,000 \times 1,000,000 \times 4\ \text{bytes} \approx 558\ \text{GB}
\end{aligned}
$$

This far exceeds the memory capacity of typical GPUs such as the H100, which has around 80 GB of memory. Storing this matrix explicitly is therefore infeasible.

As an exercise, consider how to compute the derivatives without materializing this large probability matrix.