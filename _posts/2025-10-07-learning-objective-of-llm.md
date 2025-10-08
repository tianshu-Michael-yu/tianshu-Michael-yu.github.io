## Learning Objective of LLM

The fundamental of llm is to predict next token. In pretrain and SFT, given a sentence as the ground truth,
we want the output token at each step more likely to be the token in the ground truth. Let's 
denote the sentence as $\vec{text}$. Our objective is to maximize the $P(\vec{text})$

Let's denote the sequence of token in the sentence up to the ${t-1}^{th}$ token $s_t$. 
Let's denote the $t^{th}$ token $a_t$.

$$
P(\vec{text}) = \prod P(a_t|s_t)
$$

So our goal is to maximize this product. But these probability numbers are extremely small. Multiply them together
would be close to zero and subject to the hardware round off problem. Instead, we take the log of that probability.
There're two nice thing about log. It's monotonically increasing, meaning that maximize probability is the same as
maximize its log. log turns very small number to much managerable number. So we can write our objective function in
a more manageable way.

$$
\log P(\vec{text}) = \sum \log P(a_t|s_t)
$$

Notice that this expression is negative because probability is smaller than 1. We add a negative sign before that.
So maximize the probabilty is now minimize the negative log prob. So we get our loss function.

$$
Loss  = - \sum \log P(a_t|s_t)
$$ 

### Calculate the gradient for each logit
Let's set the following notation.

$$
y_t = \log P(a_t|s_t)
p_t = P(a_t|s_t)
$$

Then we can apply chain rule.
$$
\frac{\partial Loss}{\partial y_t} = -1 \\
\frac{\partial y_t}{\partial p_t} = \frac{1}{p_t} \\
\frac{\partial Loss}{\partial p_t} = \frac{\partial Loss}{\partial y_t} \frac{\partial y_t}{\partial p_t} = -\frac{1}{p_t}
$$

I will abuse my notation a bit. From now on, we use $p(k)$ to denote $P(a_t|s_t)$ where $a_t$ is the $k^th$
token in the dictionary. $p(k)$ is the output of softmax function applied to the $k^{th}$ logit, $z(k)$.

$$
p(k) = \frac{e^{z(k)}}{\sum_j e^{z(j)}} \\
\begin{aligned}
\frac{\partial p(k)}{\partial z(i)}  &= \frac{e^{z(k)} \delta_{ik} \sum_j e^{z(j)} - e^{z(k)}e^{z(i)}}{[\sum_j e^{z(j)}]^2} \\
&=\frac{(\delta_{jk} \sum_j e^{z(j)} -e^{z(i)})e^{z(k)}}{[\sum_j e^{z(j)}]^2} \\
&= p(k) (\delta_{ik} - p(i))
\end{aligned}
$$

Use chain rule to calculate the derivative w.r.t $t^{th}$ logits.

$$
\frac{\partial Loss}{\partial z_t(i)} = \frac{\partial Loss} {\partial p_t} \frac{\partial p_t}{z_t(i)}
=-\frac{1}{p_t(k)} p_t(k) (\delta_{ik} - p_t(i))
= p_t(i) - \delta_{ik}
$$

### Update parameters
Let's update the lm_head. lm_head is usually an unbiased linear layer. The output of lm_head is the logit z. Let's denote its weight
$W$ and input $x$ 

$$
\begin{aligned}
z_t &= Wx_t \\
z_t(i) &= \sum_j W_{ij} x_{tj} \\
\frac{\partial z_t(i)}{\partial W_{ik}} &= x_{tk}
\end{aligned}
$$

So the derivative of Loss w.r.t $W$ is
$$
\begin{aligned}
\frac{\partial Loss}{\partial W_{ik}} &= \sum_t \frac{\partial Loss}{\partial z_t(i)}\frac{\partial z_t(i)}{\partial W_{ik}} \\
&= \sum_t (p_t(i)-\delta_{ik})x_{tk}

\end{aligned}
$$

