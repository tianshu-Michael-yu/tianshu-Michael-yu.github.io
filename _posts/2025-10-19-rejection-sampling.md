## Rejection Sampling – Why It Matters in LLM Training

In the previous post, [Learning Objective of LLM]({% post_url 2025-10-07-learning-objective-of-llm %}), we introduced the core training objective of language models: maximizing the probability of generating the correct text sequence. In supervised fine-tuning (SFT), this is typically done by minimizing the negative log-likelihood:

$$
\mathrm{Loss} = - \log P(\vec{\text{text}})
$$

However, in practice, we don’t just want a model that imitates text—it needs to produce *good* responses that follow human instructions, are helpful, and align with our preferences. This raises an important question:

**Where does the high-quality training data come from?**

### The Challenge: Getting Good Instruction-Following Data

To train instruction-following models, we need pairs of `<Prompt, Response>` where the response is a *good* answer to the prompt. But high-quality responses are not abundant on the internet, and manually creating them is slow and expensive.

One idea is to ask the model itself to generate responses. But if the model is not yet good, most responses it produces will be low quality. So how do we filter out only the good ones?

### Rejection Sampling: A Simple but Powerful Idea

Rejection sampling provides a straightforward solution:

1. Start with a prompt.
2. Use the model to generate multiple responses (by sampling with randomness).
3. Rank the responses by quality.
4. Select only the best one.
5. Use the selected response as supervised fine-tuning (SFT) data.

This allows us to generate higher-quality data than any single model output.

### Scaling Selection with a Preference Model

Instead of relying on humans to rank every response, we can train a *preference model* that predicts which response a human would prefer.

To train this model:
- For each prompt, we generate two responses: $\vec{\text{text}}_i$ and $\vec{\text{text}}_j$.
- A human labels which one is better.
- The preference model learns to assign higher probability to the better response.

The loss function for training the preference model is:

$$
\begin{aligned}
\mathrm{Loss} &= -hp \cdot \log P(\vec{\text{text}}_{i}) - (1-hp) \cdot \log P(\vec{\text{text}}_{j}) \\
&= -\log P(\vec{\text{text}}_{w})
\end{aligned}
$$

where $\vec{\text{text}}_w$ is the winning response.

Once trained, this preference model can automatically rank generations, enabling large-scale rejection sampling without manual labeling.


### Limitations of Rejection Sampling

1. **Inefficient and Discrete Learning**
   Rejection sampling operates in large, separate rounds of data generation and training. The model generates many responses, selects the best ones, and then performs many gradient updates on this fixed dataset. This process is inefficient because improvements to the model happen only after an entire batch of data is processed, rather than continuously. Ideally, the model should learn incrementally—improving slightly and immediately using that improvement to generate better data.

2. **No Learning from Mistakes**
   Only the best response for each prompt is kept, while all other responses are discarded. This means the model receives purely positive feedback and learns nothing from incorrect or suboptimal outputs. In tasks like math or reasoning—where trial and error is essential—ignoring the “failed attempts” wastes valuable learning signals.

3. **High Computational Cost**
   Rejection sampling requires generating multiple responses per prompt. For example, sampling 10 responses for each of 100k prompts results in 1 million forward passes, even though only 100k of those responses are used. This is extremely inefficient, especially when working with large models where inference is expensive.

4. **Risk of Model Collapse and Bias**
   By always selecting the single highest-scoring response, the model may overfit to the scoring criteria of the preference model. Over time, this can reduce diversity and lead to reward hacking—where the model learns to exploit flaws in the preference model rather than genuinely improve response quality. This can result in outputs that score well but are unhelpful or incoherent.
