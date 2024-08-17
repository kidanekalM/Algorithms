import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Iris dataset
data = np.genfromtxt('C:/Users/PC/Desktop/New folder/slides/semester 10 - spring/cs488 AI/assing/Algorithms/Iris.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1]
y[y == 'setosa'] = 0
y[y == 'versicolor'] = 1
y[y == 'virginica'] = 2
y = y.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class Perceptron:
    def __init__(self, learning_rate=0.1, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            for i in range(num_samples):
                prediction = np.dot(X[i], self.weights) + self.bias
                if (prediction * y[i]) <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)

    # Train the perceptron classifier
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Evaluate the model
y_pred = perceptron.predict(X_test)
accuracy = np.mean(y_pred == y_test)
precision = []
recall = []
f1_score = []

for i in range(3):
    tp = np.sum((y_pred == i) & (y_test == i))
    fp = np.sum((y_pred == i) & (y_test != i))
    fn = np.sum((y_pred != i) & (y_test == i))
    precision.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
    recall.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
    f1_score.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {', '.join(f'{p:.2f}' for p in precision)}")
print(f"Recall: {', '.join(f'{r:.2f}' for r in recall)}")
print(f"F1-score: {', '.join(f'{f:.2f}' for f in f1_score)}")

# Visualize the decision boundaries
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', s=60)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', s=60, marker='s')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Perceptron Decision Boundaries')
plt.legend(['Setosa', 'Versicolor', 'Virginica', 'Training Set', 'Test Set'])
plt.show()