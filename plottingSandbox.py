weights1=[1,-10, -18.499999999999982, -21.699999999999996, 29.09999999999999, 26.20000000000001]
weights2=[1,-1, -1.9000000000000004, -4.9, 6.500000000000002, 3.4999999999999996]
import matplotlib.pyplot as plt
import numpy as np

# Define the weights
# weights1 = [-10, -18.499999999999982, -21.699999999999996, 29.09999999999999, 26.20000000000001]
# weights2 = [-1, -1.9000000000000004, -4.9, 6.500000000000002, 3.4999999999999996]

# Define the activation function (ReLU in this case)
def relu(x):
    return np.maximum(0, x)

# Generate x values
x = [[]]

# Calculate y values for each set of weights
# y1 = relu(np.dot(weights1, np.ones(len(weights1))))
# y2 = relu(np.dot(weights2, np.ones(len(weights2))))
y1 = weights1[0]*x[0] + weights1[1]*x[1] + weights1[2]*x[2] + weights1[3]*x[3] + weights1[4]*x[4] + weights1[5]*x[5]
y2 = weights2[0]*x[0] + weights2[1]*x[1] + weights2[2]*x[2] + weights2[3]*x[3] + weights2[4]*x[4] + weights2[5]*x[5]
# Plot the results
plt.plot(x, y1 * np.ones_like(x), label='Weights1')
plt.plot(x, y2 * np.ones_like(x), label='Weights2')
plt.xlabel('Input')
plt.ylabel('Activation')
plt.title('Activation Function with Given Weights')
plt.legend()
plt.show()

def draw_decision_boundary(weights1, weights2,X_train):
    x = X_train.to_numpy()

    # Calculate y values for each set of weights
    # y1 = relu(np.dot(weights1, np.ones(len(weights1))))
    # y2 = relu(np.dot(weights2, np.ones(len(weights2))))
    y1 = weights1[0]*x[0] + weights1[1]*x[1] + weights1[2]*x[2] + weights1[3]*x[3] + weights1[4]*x[4] + weights1[5]*x[5]
    y2 = weights2[0]*x[0] + weights2[1]*x[1] + weights2[2]*x[2] + weights2[3]*x[3] + weights2[4]*x[4] + weights2[5]*x[5]
    # Plot the results
    plt.plot(x, y1 * np.ones_like(x), label='Weights1')
    plt.plot(x, y2 * np.ones_like(x), label='Weights2')
    plt.xlabel('Input')
    plt.ylabel('Activation')
    plt.title('Activation Function with Given Weights')
    plt.legend()
    plt.show()