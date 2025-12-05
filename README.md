# CS336 Language Modeling Project — Tiny Transformer from Scratch

This project is my personal implementation inspired by **Stanford CS336 (Language Modeling from Scratch)**.  
It builds a minimal end-to-end language model training pipeline using only basic PyTorch modules:

- ✔️ Word-level tokenizer  
- ✔️ Tiny Transformer LM (attention + MLP + positional embeddings)  
- ✔️ Training loop (checkpoint saving included)  
- ✔️ Simple evaluation (perplexity)  
- ✔️ Sampling (text generation)  

## 📂 Project Structure

CS336-Language-Modeling-Project/
│
├── src/
│ ├── tokenizer.py # Word-level tokenizer
│ ├── model.py # Tiny Transformer model
│ ├── dataset.py # Next-token dataset
│ ├── train.py # Training script
│ ├── sampling.py # Text generation
│ └── eval_ppl.py # Perplexity evaluation
│
├── data/
│ └── tiny_corpus.txt # Small training corpus
│
├── checkpoints/ # Saved model checkpoints
├── requirements.txt
├── README.md
└── LICENSE


---

## 🚀 Quick Start

### 1. Install environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2. Prepare data

Place your training text under:
data/tiny_corpus.txt
You can use any plain-English text.
🧠 Train the Tiny Transformer
python -m src.train \
  --data_path data/tiny_corpus.txt \
  --block_size 64 \
  --batch_size 32 \
  --n_epochs 5


This will:

build a vocabulary → data/vocab.json

train the LM

save checkpoints → checkpoints/epoch_*.pt

✨ Generate Text

After training, run:

python -m src.sampling \
  --checkpoint checkpoints/epoch_3.pt \
  --vocab_path data/vocab.json \
  --prompt "once upon a time" \
  --max_new_tokens 80 \
  --temperature 0.8 \
  --top_k 20


Example output:

once upon a time in a small village a traveler carried a book of stories...

📏 Evaluate Perplexity
python -m src.eval_ppl \
  --data_path data/tiny_corpus.txt \
  --checkpoint checkpoints/epoch_3.pt \
  --vocab_path data/vocab.json \
  --block_size 64 \
  --batch_size 32


Output:

Perplexity: 21.84

📓 Optional: Notebook Demo

The project supports Jupyter Notebook.
You can create a notebook under notebooks/ and reuse the modules:

from src.tokenizer import Tokenizer
from src.model import TinyTransformerLM
from src.sampling import sample

🔧 Model Architecture (Tiny)

Embedding + learned positional embeddings

Multi-head self-attention (causal mask)

Feed-forward MLP

LayerNorm + residual connections

Linear output projection to vocab logits

This small model is ideal for learning:

how attention works

how an LM predicts next tokens

how perplexity relates to LM quality

📌 Future Improvements (Optional Ideas)

Add validation split + early stopping

Implement BPE tokenizer

Add multi-layer attention visualizations

Train on larger corpora (TinyStories, WikiText)

Export to ONNX or TorchScript
