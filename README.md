# DA6401 Assignment 3 — Transformer for Neural Machine Translation (De→En)

**Student:** Ganesh Mula | **Roll No:** DA25M019 | M.Tech DS-AI

---

## W&B Report

[Click here to view the full Weights & Biases Report](https://api.wandb.ai/links/da25m019-indian-institute-of-technology-madras/s5k013sc)

---

## Overview

Implementation of the landmark architecture from **"Attention Is All You Need"** (Vaswani et al., 2017) from scratch using PyTorch. The goal is a Neural Machine Translation system for German → English translation trained on the Multi30k dataset.

---

## Project Structure

```
DA25M019_DA6401_Assignment_3/
├── model.py          # Full Transformer architecture + infer()
├── train.py          # Training loop, loss, BLEU evaluation
├── dataset.py        # Multi30k data loading and vocabulary
├── lr_scheduler.py   # Noam learning rate scheduler
└── README.md
```

---

## Architecture

The Transformer is implemented strictly following the paper with the following components:

| Component | Details |
|---|---|
| d_model | 256 |
| Encoder/Decoder Layers (N) | 3 |
| Attention Heads | 8 |
| Feed-Forward dim (d_ff) | 512 |
| Dropout | 0.1 |
| Positional Encoding | Sinusoidal (registered buffer) |
| Normalization | Post-LayerNorm |

### Key Modules

- **Scaled Dot-Product Attention** — `Attention(Q,K,V) = softmax(QKᵀ/√dk) · V`
- **Multi-Head Attention** — 4 weight matrices (Wq, Wk, Wv, Wo), no `nn.MultiheadAttention`
- **Positional Encoding** — sinusoidal, registered as non-trainable buffer
- **PositionwiseFeedForward** — `FFN(x) = max(0, xW₁+b₁)W₂+b₂`
- **Masking** — padding mask for encoder/decoder, causal look-ahead mask for decoder

---

## Training

### Dataset
- **Multi30k** (`bentrevett/multi30k` on HuggingFace)
- 29,000 train / 1,014 validation / 1,000 test pairs
- Tokenization via spaCy (`de_core_news_sm`, `en_core_web_sm`)
- Vocabulary built with `min_freq=2`

### Optimization

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| LR Schedule | Noam (warmup_steps=4000) |
| Label Smoothing | ε = 0.1 |
| Batch Size | 128 |
| Epochs | 20 |
| Gradient Clipping | 1.0 |

### Noam Schedule
```
lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

---

## Inference

The autograder calls `model.infer(german_sentence)` directly:

```python
model = Transformer().to(device)
model.eval()
english_sentence = model.infer("Ein Hund rennt durch das Gras.")
```

On instantiation, `Transformer.__init__` automatically:
1. Downloads `vocabs.pt` from Google Drive
2. Loads vocabulary and spaCy tokenizer
3. Downloads `best_checkpoint.pt` from Google Drive
4. Loads trained weights into the architecture

---

## W&B Experiments

### 2.1 Noam Scheduler vs Fixed LR
Compared Noam warmup schedule against constant LR=1e-4. The Transformer's self-attention layers initialise near zero; a large fixed LR immediately saturates the softmax, causing vanishing gradients. The warmup phase keeps updates small while attention weights stabilise.

### 2.2 Ablation: Scaling Factor 1/√dk
Without 1/√dk, raw dot-products grow as O(dk), pushing softmax into near-zero gradient regions (Section 3.2.1 of the paper). Q and K gradient norms collapse in the unscaled variant, confirming the vanishing-gradient effect.

### 2.3 Attention Head Specialisation
Per-head attention heatmaps extracted from the last encoder layer reveal distinct head behaviours — local adjacency attention, syntactic long-range dependencies, and some head redundancy.

### 2.4 Sinusoidal vs Learned Positional Encoding
Sinusoidal PE encodes position with fixed sin/cos functions, allowing theoretical extrapolation beyond training sequence lengths. Learned PE (`nn.Embedding`) can only interpolate within the training length range.

### 2.5 Label Smoothing (ε=0.1 vs ε=0.0)
Label smoothing redistributes probability mass from the correct token uniformly across the vocabulary, preventing overconfidence. Training perplexity is higher with smoothing, but generalisation improves.

---

## Results

| Test | BLEU |
|---|---|
| BLEU > 20.0 | ✅ |
| BLEU > 25.0 | ✅ |
| BLEU > 30.0 | ✅ |
| BLEU > 35.0 | 32.17 |

---

## Permitted Libraries

`torch`, `numpy`, `matplotlib`, `scikit-learn`, `wandb`, `datasets`, `spacy`, `evaluate`, `tqdm`, `gdown`

---

## References

Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
[https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
