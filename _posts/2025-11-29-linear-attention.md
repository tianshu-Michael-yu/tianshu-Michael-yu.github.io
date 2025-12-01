# Linear Attention

Let's start with some notations:

$$
q_i, k_i, v_i \in \mathbb{R}^d \\
Q = [q_1, q_2, \ldots, q_n] \in \mathbb{R}^{n \times d} \\
K = [k_1, k_2, \ldots, k_n] \in \mathbb{R}^{n \times d} \\
V = [v_1, v_2, \ldots, v_n] \in \mathbb{R}^{n \times d} \\
O = [o_1, o_2, \ldots, o_n] \in \mathbb{R}^{n \times d}
$$

The core of attention is a mapping $Q, K, V \rightarrow O$.

## Softmax Attention

The mapping is usually defined as:

$$
O = \text{softmax}(QK^\top+\log M)V
$$

$M \in \mathbb{R}^{n \times n}$ is a lower triangular matrix, called causal mask:

$$
M_{i,j} = \begin{cases}
    1 & \text{if } i \geq j \\
    0 & \text{if } i < j
\end{cases}
$$

This is our usual Softmax Attention. Written in elementwise form, it is:

$$
o_t = \frac{\sum_{j=1}^{t} \exp(q_t \cdot k_j) v_j}{\sum_{j=1}^{t} \exp(q_t \cdot k_j)}
$$

The denominator here is majorly for numerical stability. If we forget about the causal mask. The core of the attention is really $O=\exp(QK^\top)V$. The problem of softmax attention is that to compute $\exp(QK^\top)$, both space and time complexity are $O(n^2)$

## Linear Attention

The first idea of linear attention is to replace the expensive softmax computation with a kernel-based approximation of the attention weight. The kernelized linear attention aims to find a feature mapping $\phi$ such that  $\exp(q \cdot k) \approx \phi(q) \cdot \phi(k)$. Given such $\phi$, one can rewrite attention as:

$$
O = (\phi(Q) \phi(K)^\top) V = \phi(Q) (\phi(K)^\top V)
$$

If we take $\phi$ to be the identity mapping, we get $O = Q(K^\top V)$. Since $K^\top V$ is a $d \times d$ matrix, this reformulation reduce complexity from $O(n^2)$ to $O(n)$.  

We were ignoring the causal mask $M$ in the above analysis. Let's bring it back.

$$
o_t = \sum_{j=1}^{t} v_j (k_j^\top q_t) = \sum_{j=1}^{t} (v_j k_j^\top)q_t = (\sum_{j=1}^{t} v_j k_j^\top)q_t
$$

Let's denote the part inside the parenthesis as $S_t$, then:

$$
o_t = S_t q_t, \quad S_t = S_{t-1} + v_t k_t^\top
$$

This is our vanila linear attention. It's clear that the casual form of attention can be written as a linear RNN with $S_t$ as the hidden state. Every step's time complexity is constant, so the total time complexity is $O(n)$.

## Linear Attention As Test Time Training

![TTT](/img/TTT.png)

There's a beautiful conceptualization of linear attention as Test Time Training (TTT). TTT can be used to construct RNN. Here's the procdure: Let the current model's parameter be $S_{t-1}$. The optimizer (SGD) receives new data $k_t, v_t$. The optimizer updates the parameter to $S_t$ using the new data and return the prediction result $f(S_{t-1}; q_t)$. So all RNN created with TTT can be written as:

$$
o_t = f(S_t; q_t), \quad S_t = S_{t-1} - \eta_t \nabla_{S_{t-1}}\mathcal{L}(f(S_{t-1}; k_t), v_t)
$$

, where $\mathcal{L}$ is the loss function and $\eta_t$ is the learning rate. 

### Vanilla Linear Attention

Let's set $f, \mathcal{L}, \eta_t$ to be:

$$
f(S; k) = S k, \quad \mathcal{L}(y, v) = -v^\top y, \quad \eta_t = 1
$$

Let's calculate the graident of $\mathcal{L}$ with respect to $S$:

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial y_l} &= -v_l \\
\frac{\partial y_l}{\partial S_{ij}} &= \delta_{li} k_j \\
\frac{\partial \mathcal{L}}{\partial S_{ij}} &= \sum_{l=1}^{n} \frac{\partial \mathcal{L}}{\partial y_l} \frac{\partial y_l}{\partial S_{ij}} \\
&= \sum_{l=1}^{n} -v_l \delta_{li} k_j \\
&= -v_i k_j
\end{align}
$$

So in matrix form:

$$
\nabla_S \mathcal{L} = -v k^\top
$$

Plug this back into the TTT update rule, we get:

$$
o_t = S_t q_t, \quad S_t = S_{t-1} + v_t k_t^\top
$$

This shows that the vanilla linear attention is a special case of TTT.

### DeltaNet

Now we have a nice paradigm to construct RNNs. Why don't we try other loss functions?

Let the loss function be the L2 loss divided by 2:

$$
\mathcal{L}(y, v) = \frac{1}{2} ||y - v||^2
$$

The gradient of this loss function with respect to $y$ is $y-v$. Plug this back to (3), we get:

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial S_{ij}} &= \sum_{l=1}^{n} (y_l - v_l) \delta_{li} k_j \\
&= (y - v) k_j \\
\nabla_S \mathcal{L} &= (y - v) k^\top \\
&=(Sk-v)k^\top
\end{align}
$$

Plug this back in the TTT update rule and keep the learning rate to be time depedent, we get:

$$
\begin{align}
S_t &= S_{t-1} - \eta_t (S_{t-1}k_t-v_t) k_t^\top \\
&= S_{t-1}(I-\eta_t k_t k_t^\top) + \eta_t v_t k_t^\top
\end{align}
$$

This gives us the DeltaNet. 

### GatedDeltaNet and KDA

If we set the loss function to be $\frac{1}{2} ||y-v||^2 + \frac{1-\gamma}{\eta}||S||_F^2$. The update rule becomes 

$$
S_t = \gamma_t S_{t-1} + \eta_t (v_t - S_{t-1}k_t) k_t^\top
$$

If we set $\gamma_t = \alpha_t, \eta_t= \alpha_t \beta_t$, and we absorb $\alpha_t$ into $v_t$, we get:

$$
S_t = \alpha_t S_{t-1} (I-\beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

This is the GatedDeltaNet.

If we change the $\alpha_t$ to a diagonal matrix $\text{Diag}(\alpha_t)$, we get:

$$
S_t = \text{Diag}(\alpha_t) S_{t-1} (I-\beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

This is the Kimi Delta Attention (KDA) presented in the KimiLinear paper.

## Advanced TTT Variants

We don't have restrict ourself to just using different loss functions. We can also experiment with different update rules. For example, in the Titans paper, the author add a momentum term to the SGD update rule. The Test-Time Training Done Right paper has explored the possibility of using Muon optimizer instead of SGD. 

So why is TTT such a powerful paradigm for constructing linear attentions? The core goal of linear attention is just to compress historical data to a fixed sized state. A model's parameter can be also viewed as a fixed sized state. This implies that training a model is not dissimilar to compressing training data to model weight. TTT uses this insight. More formally, if linear attention is a compression task, TTT views the model $f$ as the decoder of the compression task , the weights as the compressed data, the optimizer as the compression algorithm, and the loss function $\mathcal{L}$ as the compression quality metric.
