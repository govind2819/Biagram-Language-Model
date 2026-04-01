# Bigram Language Model (PyTorch)

## Overview

This project implements a character-level Bigram Language Model from scratch using PyTorch.
It includes both a statistical approach and a neural network-based approach to model character transitions and generate new names.

The goal of this project is to understand the fundamentals of language modeling, probability, and gradient-based learning.

---

## Features

* Bigram frequency matrix construction
* Probability estimation using normalization
* Log-likelihood and Negative Log-Likelihood (NLL)
* Laplace smoothing
* Neural bigram model using one-hot encoding
* Training using gradient descent
* Character-level name generation

---

## Tech Stack

* Python
* PyTorch
* Matplotlib

---

## Dataset

The model is trained on a dataset of names (`names.txt`).
Each name is processed at the character level, including special start (`.`) and end (`.`) tokens.

---

## Model Approach

### Statistical Bigram Model

* Counts character transitions
* Converts counts into probabilities
* Evaluates model using log-likelihood

### Neural Bigram Model

* One-hot encodes input characters
* Learns transition probabilities using a weight matrix
* Optimized using cross-entropy loss (equivalent to NLL)

---

## Training

The neural model is trained using gradient descent:

* Forward pass: `logits = xenc @ W`
* Loss function: Cross-entropy
* Backpropagation: `loss.backward()`
* Weight update: Gradient descent

---

## Results

* Statistical Bigram Model NLL ≈ 2.2
* Neural Bigram Model converges to similar performance

---

## Sample Output

Example generated names:

```
anaya
rohit
devansh
riya
karan
```

---

## Project Structure

```
bigram-language-model/
│
├── makemore.ipynb
├── makemore.py
├── names.txt
├── README.md
├── requirements.txt
```

---

## How to Run

1. Clone the repository
2. Install dependencies:

   ```
   pip install torch matplotlib
   ```
3. Open the notebook:

   ```
   jupyter notebook
   ```
4. Run all cells

---

## Learning Outcomes

* Understanding of bigram language models
* Implementation of probability-based models
* Hands-on experience with PyTorch tensors and gradients
* Debugging neural network training issues
* Building models from scratch

---

## Future Improvements

* Extend to trigram or n-gram models
* Implement MLP-based language model
* Build mini GPT / transformer architecture

---

## Inspiration

This project is inspired by the teachings of Andrej Karpathy and his deep learning lectures.

---

## Author

Govind Solanki
