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

Quick Start

Step 1: Create environment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Step 2: Prepare data
Place your training file at:
data/tiny_corpus.txt

You may use any plain English text.

Training

Run the training script:

python -m src.train
--data_path data/tiny_corpus.txt
--block_size 64
--batch_size 32
--n_epochs 5

This will:

Build a vocabulary file at data/vocab.json

Train a tiny Transformer model

Save checkpoints under checkpoints/

Text Generation

Use a trained checkpoint to generate text:

python -m src.sampling
--checkpoint checkpoints/epoch_3.pt
--vocab_path data/vocab.json
--prompt "once upon a time"
--max_new_tokens 80
--temperature 0.8
--top_k 20

Example output:
once upon a time in a small village a traveler carried a book of stories...

Perplexity Evaluation

Compute perplexity on a text file:

python -m src.eval_ppl
--data_path data/tiny_corpus.txt
--checkpoint checkpoints/epoch_3.pt
--vocab_path data/vocab.json
--block_size 64
--batch_size 32

Example:
Perplexity: 21.84

Model Architecture

Embedding layer with learned positional embeddings
Multi-head self-attention (causal mask)
Feed-forward MLP
LayerNorm and residual connections
Linear output projection to vocab logits

This project helps understand:
How attention works
How next-token prediction functions
How perplexity measures language model quality

Future Improvements

Optional extensions:
Add validation split and early stopping
Use BPE or subword tokenizer
Visualize attention maps
Train using larger corpora such as TinyStories or WikiText
Export model to ONNX or TorchScript
