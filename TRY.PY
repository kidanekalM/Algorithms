import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def split_iris_and_plot_decision_boundary():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a perceptron model
    w = np.zeros(X_train.shape[1] + 1)  # Initialize weights
    learning_rate = 0.1
    epochs = 100

    for _ in range(epochs):
        for i in range(len(X_train)):
            x = np.concatenate((np.array([1]), X_train[i]))
            y_pred = 1 if np.dot(w, x) > 0 else 0
            w = w + learning_rate * (y_train[i] - y_pred) * x

    # Make predictions on the test set
    y_pred = []
    for i in range(len(X_test)):
        x = np.concatenate((np.array([1]), X_test[i]))
        y_pred.append(1 if np.dot(w, x) > 0 else 0)

    # Plot the decision boundaries
    plot_decision_boundary(w, X_test, y_test, y_pred)
def plot_decision_boundary(weights, X_test, y_test, pred_array):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the function value for the whole grid
    grid = np.column_stack((np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()))
    Z = np.dot(grid, weights)
    Z = np.where(Z > 0, 1, 0)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o', s=100, label='Test Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_array, edgecolors='k', marker='x', s=100, label='Predictions')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.title('Decision Boundary')
    plt.show()
# Run the function to split the Iris dataset and plot the decision boundaries
split_iris_and_plot_decision_boundary()