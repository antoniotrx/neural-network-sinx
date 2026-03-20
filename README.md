# Calculating sin(x) with a neural network

A neural network built from scratch in Python that learns to aproximate the function y=sin(x)

📄 [View full documentation (PDF)](./documentation.pdf)

## Overview
The network is trained on angle/sine value pairs and learns to predict sin(x) for any input angle.
 - **Architecture**: 1 input neuron → 5 hidden neurons → 1 output neuron
 - **Activation**: A scaled sigmoid function to match the range of sin(x)
 - **Training**:Gradient descent with backpropagation algorithm.

## Results
![Comparison Plot: Predicted VS True Values](plot.png)

The network approximates sin(x) well in [0, 360] but doesn't generalize beyond it.

### Requirements
pip install numpy matplotlib
