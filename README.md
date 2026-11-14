# English-to-Spanish Machine Translation

## Overview
This project implements an English-to-Spanish neural machine translation system using a Transformer encoder-decoder architecture built with PyTorch. The model is trained from scratch using parallel English and Spanish sentence pairs.

## Features
- Transformer-based encoder-decoder architecture.
- English and Spanish vocabulary processing.
- GPU acceleration using NVIDIA CUDA.
- Training and evaluation using BLEU scores.
- Translation of English sentences into Spanish.

## Requirements
- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- SacreBLEU

Install dependencies:

```bash
pip install torch numpy matplotlib sacrebleu==2.6.0
```

## Usage

1. Open `assignment4_translation.ipynb`.
2. Select your Python environment or GPU-enabled kernel.
3. Run the notebook cells to preprocess the dataset and train the model.
4. Evaluate translation performance using BLEU scores.
5. Use the trained model to translate English sentences into Spanish.

## GPU Support
The project supports NVIDIA GPU acceleration through PyTorch CUDA.

Verify GPU availability:

```python
import torch
print(torch.cuda.is_available())
```

## Evaluation
Translation performance is measured using the BLEU score through SacreBLEU. Higher BLEU scores indicate closer agreement with reference translations.

## Technologies
- Python
- PyTorch
- Transformer Neural Networks
- CUDA
- SacreBLEU