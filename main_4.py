import math
import numpy as np
import matplotlib.pyplot as plt
from main_2 import w1, b1, w2, b2

#print(f"w^[1] = {w1}") # Weight coeficients for connections between input and hidden layer
#print(f"b^[1] = {b1}") # Bias coeficients for hidden layer
#print(f"w^[2] = {w2}") # Weight coeficients for connections between hidden and output layer
#print(f"b^[2] = {b2}") # Bias coeficient for output layer

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def scaled_sigmoid(sigmoid):
    return 2*(sigmoid) - 1

# Example
test_angles = [0, 5, 18, 30, 45, 67, 90, 120, 150, 180, 220, 270, 310, 360]
outputs_neural_network = []
true_values = []

for angle_deg in test_angles:
    x_rad = math.radians(angle_deg)
    y = math.sin(x_rad) # y = y(x) = sin(x)
    true_values.append(y)

    z_2 = w1 * x_rad + b1
    sigma_2 = sigmoid(z_2)
    layer_2 = scaled_sigmoid(sigma_2)

    z_3 = np.dot(np.transpose(w2), layer_2) + b2
    sigma_3 = sigmoid(z_3)
    output = scaled_sigmoid(sigma_3)[0][0] # Predicted value
    outputs_neural_network.append(output)

    abs_err = np.abs(output - y)
    cost = 0.5 * math.pow(output - y, 2)

    #print(f"Neural Network value = {output}")
    #print(f"True value = {y}")
    #print(f"Absolute error = {abs_err}")
    #print(f"Cost function = {cost}")

print("Angle, Prediction, True value")
for i in range(len(test_angles)):
    print(test_angles[i], outputs_neural_network[i], true_values[i])

plt.plot(test_angles, outputs_neural_network)
plt.plot(test_angles, true_values)
plt.savefig("plot.png")