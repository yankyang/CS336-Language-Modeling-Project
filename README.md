# CS336: Language Modeling from Scratch (Stanford, Spring 2025)

**Instructor:** Prof. Percy Liang, Prof. Tatsunori Hashimoto  
**Course Website:** [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)  
**Author:** Yankai Yang ([@yankykyang](https://github.com/yankykyang))  

---

## 🧠 Overview

This repository contains my course project for **Stanford CS336: Language Modeling from Scratch (Spring 2025)**.  
The project aims to build, train, and evaluate a transformer-based language model from scratch, while exploring how model architecture, data scale, and reasoning depth affect performance and generalization.

---

## 🎯 Project Objectives

- Implement a **minimal yet extensible transformer language model** without relying on high-level frameworks.  
- Train the model on curated subsets of **WikiText** and **OpenWebText** to study scaling behavior and perplexity trends.  
- Experiment with **tokenization strategies** (BPE, byte-level, unigram) and evaluate their impact on convergence and loss.  
- Conduct ablation studies on:
  - Model depth and number of attention heads  
  - Context window size  
  - Learning rate schedule and optimizer choice  
- Visualize loss curves, gradient norms, and performance trade-offs.  
- (Future work) Extend to reasoning-augmented or alignment-aware variants, connecting with research directions such as **ReasAlign**.

---

## 🧩 Repository Structure

