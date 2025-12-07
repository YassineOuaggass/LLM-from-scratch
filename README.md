# Build a Large Language Model (From Scratch)

This repository contains my implementation of a **GPT-like large language model (LLM)**, built from scratch. The project is based on **Build a Large Language Model (From Scratch)** by Sebastian Raschka, where I walk through the process of designing, training, and fine-tuning a transformer-based model using PyTorch.


## Repository Structure

### Core Architecture
* `model.py`: Contains the main GPT architecture (Multi-head attention, FeedForward, Transformer Blocks).
* `data_loader.py`: Handles text tokenization and batch generation.
* `load_weights.py` & `gpt_download.py`: Utilities for loading pretrained weights into the custom architecture.

### Training & Fine-Tuning
* `train.py` / `train.ipynb`: Scripts for pretraining the LLM on `the-verdict.txt`.
* `classifier_finetuning.ipynb`: Notebook demonstrating fine-tuning for spam classification using `SMSSpamCollection.tsv`.
* `instruction_following_finetune.ipynb`: Notebook for fine-tuning the model to follow instructions (Alpaca-style) using `instruction-data.json`.


## Key Highlights:
- **End-to-End Model Pipeline**: From building the attention mechanism to implementing the full GPT model architecture.
- **Pretraining & Fine-Tuning**: Implemented pretraining on large text datasets and fine-tuning for specific tasks, such as text classification and instruction-following.
- **PyTorch Implementation**: The entire codebase is built with PyTorch, demonstrating in-depth knowledge of deep learning and neural network architectures.
- **Hands-On Approach**: This project reflects practical experience in implementing complex models from scratch.

## Project Overview:
This work demonstrates the ability to create a GPT-like LLM, showcasing both technical and problem-solving skills. The implementation is aligned with state-of-the-art techniques in the field, using a hands-on approach to understand LLMs deeply.

**Book:** [Build a Large Language Model (From Scratch) - Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)  
**GitHub Repository:** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
