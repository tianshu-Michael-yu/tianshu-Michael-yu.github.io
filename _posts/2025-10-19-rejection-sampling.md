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

### Problems of rejection sampling

1. Discrete learning: We are learning in discrete steps where we sample a lot of responses on a lot of prompts and run a round of SFT from improved data. This means we run a lot of parameter upadate over the rejection-sampled data. In theory, our model can do a few small update after we get some good data. And immediately use the slightly improved model to get much better data. That should be more efficient than doing many updates over a large dataset.

2. No learning from mistakes. We essentially learn nothing from those bad output. The model only gets signal from the best answer. The feedback is purely positive. Model trained this way will lack the capability of correct itself when it starting with a bad responses. This is especially true for math reasoning, where starting with the right reasoning is rare and it almost always involves try and error.

3. High Computation Cost: Generating many samples per prompt is wasteful, espeically for large models. If we generate 10 candidates for each of 100k prompts, that's 1 million model forward passes to sift out 100k best samples.

4. Model Collapse and Bias: If we always pick the single highest-scoring answer according to a fixed criterion, we risk overoptimizing the model on that criterion. The model might start giving very narrow, optimized answer that the scorer loves but lack diversity or even coherence. Repeatedly fine-tuning on only top outputs pusing the model into areas where the scoring model is not well calibrate, causing garbage output that the scorer misteaknely rates high. This is very akin to reward hanking, where the model exploit weaknesses in the scoring system. 
