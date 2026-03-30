# LLaVA-KD: Distilling Multimodal Large Language Models

Multimodal large language models (MLLMs) like LLaVA combine a vision encoder with a language model to handle image-text tasks. The performance scales with the size of the language backbone, but so do the compute and memory requirements. LLaVA-KD (Cai et al., ICCV 2025) asks a natural question: can we transfer the capabilities of a large MLLM into a small one through knowledge distillation, rather than training the small model from scratch?

The answer turns out to be yes, but the distillation requires more care than simply applying KL divergence on output logits. LLaVA-KD introduces a three-stage training pipeline and two distillation objectives that operate on different parts of the model's representations. This post walks through the framework, examines the loss functions, and highlights some implicit assumptions in the design.

## The Architecture

Both the teacher (large MLLM, or l-MLLM) and the student (small MLLM, or s-MLLM) share the same high-level architecture: a vision encoder, a connector (MLP projector), and a language model. The vision encoder is a SigLIP model (siglip-so400m-patch14-384) shared across all variants. The connector maps visual features from the encoder's output space into the LLM's embedding space. The language models differ in size but come from the same family — for example, Qwen2.5-3B as the teacher and Qwen2.5-0.5B as the student.

The image processing pipeline works as follows. An image is passed through the vision encoder to produce a sequence of patch features. These features are projected by the connector into the same dimensionality as the LLM's token embeddings. The projected visual tokens are then interleaved with the text token embeddings at the position of a special image placeholder token. The LLM processes this combined sequence autoregressively, attending over both visual and textual tokens.

## The Three-Stage Training Pipeline

LLaVA-KD organizes training into three stages. Each stage has a specific purpose and uses a different combination of losses.

### Stage 1: Distilled Pre-Training (DPT)

The first stage aligns visual and linguistic representations in the student model. Despite the name "pre-training," this is not unsupervised next-token prediction over raw text. The data consists of image-caption pairs, where the model receives an image plus a text prompt, and the target is the caption or description. The prompt and visual tokens are masked out (set to the ignore index $-100$), and only the caption tokens receive supervision.

During DPT, the student trains with a combination of the standard cross-entropy loss on ground-truth caption tokens and a distillation loss that aligns the student's output distribution with the teacher's. The cross-entropy loss serves as an anchor: it ensures the student learns the correct mapping from visual inputs to textual outputs, not just a smoothed approximation of the teacher's predictions. The teacher model is imperfect, and its soft distribution may spread probability mass over plausible-but-wrong tokens. Without the hard-label grounding, the student would faithfully replicate these errors.

### Stage 2: Supervised Fine-Tuning (SFT)

The second stage is standard SFT on instruction-following data — image-question-answer triples. This stage equips the student with the ability to follow instructions and produce structured responses. No distillation loss is applied here; the model trains with cross-entropy on the response tokens alone.

### Stage 3: Distilled Fine-Tuning (DFT)

The third stage reintroduces distillation after SFT. The student has now learned the task format, so the distillation objective can refine its knowledge without interfering with instruction-following capability. This stage applies the full suite of distillation losses described below.

## Multimodal Distillation (MDist)

MDist transfers the teacher's output distribution to the student at the logit level. The implementation uses KL divergence between the teacher's and student's softmax distributions, applied in two places.

### Response-Token Distillation

For the response portion of the sequence (tokens where the label is not $-100$), the loss is a forward KL divergence:

$$
\mathcal{L}_{\text{resp}} = D_{\text{KL}}\!\left( \sigma(\mathbf{z}^T) \;\|\; \sigma(\mathbf{z}^S) \right) = \sum_k \sigma(\mathbf{z}^T)_k \log \frac{\sigma(\mathbf{z}^T)_k}{\sigma(\mathbf{z}^S)_k}
$$

where $\sigma(\mathbf{z}^T)$ and $\sigma(\mathbf{z}^S)$ are the softmax distributions of the teacher and student logits respectively, and $k$ ranges over the vocabulary. This loss is computed only over masked response positions, meaning the student is trained to match the teacher's full predictive distribution — not just the argmax — on the tokens that matter for downstream performance.

### Visual-Position Distillation

A clarification on terminology is useful here. "Visual tokens" are not part of the text vocabulary — they are continuous embeddings produced by the vision encoder and projected into the LLM's embedding space by the connector. The LLM cannot sample or output a visual token. At every sequence position, including positions occupied by visual embeddings, the LLM produces logits over the standard text vocabulary. During normal training these positions are masked ($-100$) and do not contribute to the cross-entropy loss.

However, the teacher's text-vocabulary logits at visual positions still encode useful information about how the teacher processes visual inputs internally. MDist exploits this by applying KL divergence on the LLM's output logits at these positions:

$$
\mathcal{L}_{\text{vis}} = D_{\text{KL}}\!\left( \sigma(\mathbf{z}^T_{\text{img}}) \;\|\; \sigma(\mathbf{z}^S_{\text{img}}) \right)
$$

where $\mathbf{z}^T_{\text{img}}$ and $\mathbf{z}^S_{\text{img}}$ are the teacher's and student's text-vocabulary logits at the 728 sequence positions corresponding to the SigLIP output patches. This encourages the student's language model to develop similar internal processing patterns for visual information, even at positions where no ground-truth text supervision exists.

## Relation Distillation (RDist)

RDist captures a different kind of knowledge: the structural relationships among visual positions. Rather than aligning individual output distributions, it aligns the pairwise interaction pattern across all positions occupied by visual embeddings.

Given the teacher's text-vocabulary logits at the visual positions $\mathbf{Z}^T_{\text{img}} \in \mathbb{R}^{N_v \times V}$ and the student's $\mathbf{Z}^S_{\text{img}} \in \mathbb{R}^{N_v \times V}$ (where $N_v = 728$ is the number of visual positions and $V$ is the vocabulary size), the relation matrices are:

$$
\mathbf{R}^T = \mathbf{Z}^T_{\text{img}} (\mathbf{Z}^T_{\text{img}})^\top, \qquad \mathbf{R}^S = \mathbf{Z}^S_{\text{img}} (\mathbf{Z}^S_{\text{img}})^\top
$$

Each entry $\mathbf{R}_{ij}$ is the dot product between the logit vectors at positions $i$ and $j$, capturing how similarly the model treats two visual patches in output space. The distillation loss is the cosine distance between the flattened relation matrices:

$$
\mathcal{L}_{\text{rela}} = 1 - \cos\!\left( \text{vec}(\mathbf{R}^T),\; \text{vec}(\mathbf{R}^S) \right)
$$

This loss does not require the absolute magnitudes of the logits to match — only their relative geometry. Two patches that the teacher considers similar (high dot product) should also be considered similar by the student, and vice versa. This is useful because the teacher and student have different capacity, so their raw logit scales may differ, but the relational structure can be preserved.

## The Combined Objective

During the distillation stages (DPT and DFT), the total loss for a sample containing one image is:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{resp}} + \mathcal{L}_{\text{vis}} + \mathcal{L}_{\text{rela}}
$$

where $$\mathcal{L}_{\text{CE}}$$ is the standard autoregressive cross-entropy loss. For text-only samples (no image), only $$\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{resp}}$$ is used, since there are no visual tokens to distill.

## The Shared Tokenizer Assumption

An important implicit assumption in LLaVA-KD is that the teacher and student must share the same tokenizer and vocabulary. This is not stated explicitly in the paper, but it is a hard requirement of the implementation.

The logit-level KL divergence computes $D_{\text{KL}}(\sigma(\mathbf{z}^T) \| \sigma(\mathbf{z}^S))$ where both $\mathbf{z}^T$ and $\mathbf{z}^S$ have dimension $V$ (the vocabulary size). Each index $k$ in the sum $\sum_k$ must refer to the same token for the comparison to be meaningful. If the teacher used one tokenizer and the student another, index 1042 might mean "the" for one model and "cat" for the other — the KL divergence would be nonsensical.

The code confirms this: both models receive identical `input_ids` prepared by a single tokenizer (the student's), and the trainer checks `masked_teacher_logits.shape == masked_student_logits.shape` before computing the loss. If the shapes mismatch, it falls back to the CE loss alone, effectively skipping distillation.

In practice, this means the teacher and student must come from the same LLM family. The released checkpoints use Qwen2.5-3B $\to$ Qwen2.5-0.5B and Qwen1.5 variants, all of which share the same tokenizer. If you wanted to distill from, say, a LLaMA-based teacher to a Qwen student, you would need to replace the logit-level losses with something tokenizer-agnostic, such as MSE on projected hidden states.

## Relation to Standard Knowledge Distillation

It is worth comparing LLaVA-KD's approach with the original knowledge distillation framework from Hinton et al. (2015). In the classic setup, the loss is:

$$
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, \sigma(\mathbf{z}^S)) + \alpha \cdot T^2 \cdot D_{\text{KL}}\!\left( \sigma(\mathbf{z}^T / T) \;\|\; \sigma(\mathbf{z}^S / T) \right)
$$

where $T$ is a temperature parameter that softens both distributions and $\alpha$ balances the two terms. LLaVA-KD simplifies this: there is no temperature scaling (equivalently, $T = 1$) and the losses are summed with equal weight ($\alpha = 0.5$ effectively). The key innovations are the visual-token distillation and the relational loss, neither of which exist in classical KD, since standard distillation assumes a single output per sample rather than a sequence of heterogeneous tokens mixing visual and textual modalities.

## Results

LLaVA-KD-1B (using Qwen2.5-0.5B as the student backbone) outperforms several much larger models including BLIP2-13B and InstructBLIP-7B across five multimodal benchmarks. The 1.8B variant further closes the gap with the 3B teacher. The distillation stages contribute complementary gains: DPT improves the base representations, SFT teaches the task format, and DFT refines the student's knowledge with the teacher's soft labels.

The relational distillation loss (RDist) provides consistent improvements over MDist alone, suggesting that the pairwise structure among visual tokens carries information that pointwise logit matching misses. This makes sense intuitively — understanding that "the dog's ear" and "the dog's tail" are related (both parts of the same object) is different from correctly predicting the next token given either patch.
