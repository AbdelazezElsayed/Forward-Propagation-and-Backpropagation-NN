# Feed Forward Neural Network with Backpropagation

This repository contains a Feed Forward Neural Network implemented in Python within a Jupyter Notebook. It has 2 inputs (I1, I2), 2 hidden neurons (h1, h2), and 2 outputs (o1, o2), using the `sigmoid` activation function. Weights are randomly chosen each run from [-0.5, 0.5], with biases set at 0.5 (hidden layer) and 0.7 (output layer). Built from scratch using only Python's `math` and `random` modules, it demonstrates forward propagation and backpropagation for training.

## Features
- 2 inputs, 2 hidden neurons, 2 outputs
- `sigmoid` activation with manually implemented derivative
- Random weights in [-0.5, 0.5]
- Fixed biases: 0.5 (hidden), 0.7 (output)
- Backpropagation to adjust weights and biases

## Usage
1. Open `FF-BP NN.py` in VS Code.
2. Run all cells to train the network with example inputs (0.1, 0.2) and targets (0.5, 0.3).
3. Outputs show weights, biases, inputs, and final results after 10 epochs.

## Requirements
- Python 3.x
- Jupyter Notebook

## Purpose
Created for educational purposes to demonstrate neural network training with forward and backward propagation.
