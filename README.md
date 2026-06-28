# FinNLP: Word-Level RNN Language Model (Shakespeare Corpus)

This repository contains an end-to-end Recurrent Neural Network (RNN) language model pipeline implemented from scratch using PyTorch. The system reads, tokenizes, and processes the raw *TinyShakespeare* dataset at a **word-level** (rather than character-level) to construct a robust generative model capable of producing autoregressive text extensions.

---

## Pipeline Architecture

The pipeline is split into isolated execution layers covering data ingestion, custom dataset slicing, recurrent sequence processing, and dynamic temperature-scaled inference.

* **Data Source**: Fetches the raw TinyShakespeare corpus dynamically from Karpathy's `char-rnn` distribution repository.
* **Tokenization Strategy**: Implements a word-level tokenizer using Regular Expressions (`\w+|[^\w\s]`). This separates alphanumeric words from individual punctuation marks as distinct structural tokens, giving the network a clearer structural grasp of vocabulary dependencies.
* **Lookup Mapping**: Constructs lookup dictionaries for continuous encoding and decoding:
  * `stoi` (String-to-Index): Maps word/punctuation tokens to zero-indexed integers.
  * `itos` (Index-to-String): Decodes numerical indices back to text format during text generation.

---

## Dataset & Mini-Batch Loading

The continuous tokenized corpus is transformed into overlapping historical context sequences using a custom PyTorch dataset module:

* **Sequence Slicing (`ShakespeareWordDataset`)**: Translates raw text windows into matching pairs where the input ($X$) is a chunk of history tokens, and the target ($Y$) is the exact same chunk shifted forward by one time step.
* **Context Bounds**: Structured with a context window length (`seq_len`) of **30 tokens**.
* **Loader Configurations**: The pipeline wraps the slices into a `DataLoader` with `batch_size = 64` and `drop_last=True` enabled to maintain uniform tensor dimensions during parallel processing.

---

## Neural Network Architecture (`WorldLSTM`)

The network uses a deep multi-layer Long Short-Term Memory (LSTM) structure to combat the vanishing gradient problem inherent in standard recurrent models:

* **Embedding Layer**: An `nn.Embedding` layer mapping the categorical vocabulary size into a continuous dense space.
* **Recurrent Layer**: A multi-layer `nn.LSTM` configured with `batch_first=True` to process tensor shapes natively as `(Batch Size, Sequence Length, Embedding Dimension)`.
* **Linear Output Layer**: An `nn.Linear` classifier acting as a projection head to map the hidden representations back into total vocabulary probability logits.

### Architectural Hyperparameters

| Hyperparameter | Operational Value | Structural Description |
| --- | --- | --- |
| **Embedding Dimension** | 128 | Vector dimension for dense token placement |
| **Hidden Dimension** | 256 | Internal memory state channel capacity |
| **LSTM Layer Count** | 2 | Stacked recurrent layers to extract complex dependencies |
| **Learning Rate** | 0.001 | Parameter adjustment boundary for optimization |
| **Training Epochs** | 10 | Complete optimization iterations over the corpus |

---

## Optimization & Stability Implementation

* **Loss Tracking**: Evaluates token classification errors over long tensor allocations using `nn.CrossEntropyLoss()`, flattening matrices via `.reshape(-1, vocab_size)` to pass PyTorch dimensional checks.
* **Weight Adjustment**: Driven by the `torch.optim.Adam` optimization algorithm.
* **Exploding Gradient Mitigation**: Implements standard gradient clipping via `torch.nn.utils.clip_grad_norm_` with a `max_norm=1.0` ceiling to ensure network stability over dense sequences.

---

## Autoregressive Inference Engine

The pipeline includes a custom text generation interface (`generate_text`) that implements temperature-controlled token selection:

1. **Prompt Initialization**: Processes an out-of-sample prompt string, tokenizes it, and feeds its token matrix into the network to establish the initial hidden state memory.
2. **Temperature Scaling**: Controls output creativity by scaling logits prior to processing:
   * **Higher Temperature**: Spreads out probability distributions, increasing output variety and risk.
   * **Lower Temperature**: Sharpens the distribution, forcing the engine to stick to high-confidence historical text choices.
3. **Multinomial Sampling**: Uses `torch.multinomial` to sample tokens based on scaled probabilities.
4. **Punctuation Formatting Layer**: Formats output strings to remove unnecessary whitespaces before trailing punctuation markers (such as `.,;:!?)]}`).

---

## Quickstart Implementation

### Model Definition

```python
import torch
import torch.nn as nn

class WorldLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)                     # (B, T) -> (B, T, Embed_Dim)
        out, hidden = self.rnn(x, hidden)         # (B, T, Embed_Dim) -> (B, T, Hidden_Dim)
        logits = self.fc(out)                     # (B, T, Hidden_Dim) -> (B, T, Vocab_Size)
        return logits, hidden
```

Text Generation Execution

```
# Launch target generation using out-of-sample prompt blocks
assignment_prompt = "To be or not to be, this is the problem."

generated_text = generate_text(
    model, 
    prompt=assignment_prompt, 
    max_new_tokens=100, 
    temperature=0.8
)
print(generated_text)
```

Author: Samuel Antonio Cañedo Pérez

Course Affiliation: NLP Class Spring 2026
