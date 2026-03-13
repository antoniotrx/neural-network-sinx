import math
import numpy as np

np.random.seed(4) # initialize random seed

# Neural Net Design
num_of_layers = 3
hidden_layer_size = 5

# input layer -> hidden layer -> output layer
# 1 neuron -> 5 neurons -> 1 neuron

# Example
x_deg = 45
x_rad = math.radians(x_deg)
y = math.sin(x_rad) # y = y(x) = sin(x)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def scaled_sigmoid(sigmoid):
    return 2*(sigmoid) - 1

# Parameter lists initializations
w1 = np.random.random(size=(5,1)) # 5x1
w2 = np.random.random(size=(5, 1)) # 5x1
b1 = np.random.random(size=(5, 1)) # 5x1
b2 = np.random.random() # 1x1

#print(f"w^[1] = {w1}") # Weight coeficients for connections between input and hidden layer
#print(f"w^[2] = {w2}") # Weight coeficients for connections between hidden and output layer
#print(f"b^[1] = {b1}") # Bias coeficients for hidden layer
#print(f"b^[2] = {b2}") # Bias coeficient for output layer

# Backpropagation
def backpropagation(a2, sigma2, sigma1, w2, input_value, predicted_value, true_value):
    A = 2 * sigma2 * (1 - sigma2) * (predicted_value - true_value)
    B = 2 * sigma1 * (1 - sigma1)
    grad_weight_coef_layer_3 = a2 * A
    grad_bias_layer_3 = A
    grad_weight_coef_layer_2 = w2 * B * input_value * A
    grad_bias_layer_2 = w2 * B * A
    
    return [grad_weight_coef_layer_2, grad_bias_layer_2, grad_weight_coef_layer_3, grad_bias_layer_3]

for i in range(10000):
    z_2 = w1 * x_rad + b1
    sigma_2 = sigmoid(z_2)
    layer_2 = scaled_sigmoid(sigma_2)

    z_3 = np.dot(np.transpose(w2), layer_2) + b2
    sigma_3 = sigmoid(z_3)
    output = scaled_sigmoid(sigma_3)[0][0] # Predicted value

    cost = 0.5 * math.pow(output - y, 2)

    gradient = backpropagation(layer_2 , sigma_3, sigma_2, w2, x_rad, output, y)
    grad_weight_coef_layer_2 = gradient[0]
    grad_bias_layer_2 = gradient[1]
    grad_weight_coef_layer_3 = gradient[2]
    grad_bias_layer_3 = gradient[3]

    w1 = w1 - 0.01 * grad_weight_coef_layer_2
    b1 = b1 - 0.01 * grad_bias_layer_2
    w2 = w2 - 0.01 * grad_weight_coef_layer_3
    b2 = b2 - 0.01 * grad_bias_layer_3

    if i % 500 == 0:
        print(f"Iteration {i}, Cost = {cost}")

print(f"Predicted value = {output}")
print(f"True value = {y}")
print(f"Cost value = {cost}")