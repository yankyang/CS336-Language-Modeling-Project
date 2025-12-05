CS336 Language Modeling Project — Tiny Transformer from Scratch

This project is my personal implementation inspired by Stanford CS336 (Language Modeling from Scratch).
It builds a minimal end-to-end language model training pipeline using only basic PyTorch modules:

Word-level tokenizer

Tiny Transformer LM (attention + MLP + positional embeddings)

Training loop (checkpoint saving included)

Simple evaluation (perplexity)

Sampling (text generation)

Notebook-friendly structure for experiments

This project is for self-study and portfolio purposes.
It is not an official assignment submission.

Project Structure

CS336-Language-Modeling-Project/
src/
tokenizer.py – Word-level tokenizer
model.py – Tiny Transformer model
dataset.py – Next-token dataset
train.py – Training script
sampling.py – Text generation
eval_ppl.py – Perplexity evaluation
data/
tiny_corpus.txt – Small training corpus
checkpoints/ – Saved model checkpoints
requirements.txt
README.md
LICENSE

Quick Start

Install environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Prepare data:

Put any English text into:
data/tiny_corpus.txt

Train the Tiny Transformer

python -m src.train
--data_path data/tiny_corpus.txt
--block_size 64
--batch_size 32
--n_epochs 5

This will:
• Build vocabulary → data/vocab.json
• Train the LM
• Save checkpoints → checkpoints/epoch_*.pt

Generate Text

python -m src.sampling
--checkpoint checkpoints/epoch_3.pt
--vocab_path data/vocab.json
--prompt "once upon a time"
--max_new_tokens 80
--temperature 0.8
--top_k 20

Example output:

once upon a time in a small village a traveler carried a book of stories...

Evaluate Perplexity

python -m src.eval_ppl
--data_path data/tiny_corpus.txt
--checkpoint checkpoints/epoch_3.pt
--vocab_path data/vocab.json
--block_size 64
--batch_size 32

Output:

Perplexity: 21.84

Optional: Notebook Demo

You can create Jupyter notebooks using:

from src.tokenizer import Tokenizer
from src.model import TinyTransformerLM
from src.sampling import sample

Use notebooks for interactive experiments, forward pass visualization, or text generation.

Model Architecture (Tiny)

Embedding + learned positional embeddings

Multi-head self-attention with causal mask

Feed-forward MLP

LayerNorm + residual connections

Linear output projection to vocabulary logits

This tiny model highlights:

How attention works

How next-token prediction behaves

How perplexity measures LM quality

Future Improvements

Add validation split + early stopping

Implement BPE tokenizer

Train on larger corpora (TinyStories, WikiText)

Visualize attention maps

Add unit tests and benchmarks

License

MIT License.

Acknowledgements

Inspired by Stanford CS336 “Language Modeling from Scratch”.
