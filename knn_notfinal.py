import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter

# Load the Iris dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")

    def _get_neighbors(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        sorted_indices = np.argsort(distances)[:self.k]
        return self.y_train[sorted_indices]

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            neighbors = self._get_neighbors(x)
            most_common = Counter(neighbors).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Define the values of k and distance metrics to try
k_values = [1, 3, 5, 7, 9]
distance_metrics = ['euclidean', 'manhattan']

results = []

for metric in distance_metrics:
    for k in k_values:
        knn = KNN(k=k, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
        results.append({'k': k, 'metric': metric, 'accuracy': accuracy, 
                        'precision': precision, 'recall': recall, 'f1': f1})

# Convert the results to a DataFrame for easy analysis
results_df = pd.DataFrame(results)

def plot_decision_boundary(knn, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.show()

for metric in distance_metrics:
    for k in k_values:
        knn = KNN(k=k, distance_metric=metric)
        knn.fit(X_train[:, :2], y_train)
        plot_decision_boundary(knn, X_train[:, :2], y_train, f'Decision Boundary for k={k}, metric={metric}')

# Plotting the results
metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for dist_metric in distance_metrics:
        subset = results_df[results_df['metric'] == dist_metric]
        plt.plot(subset['k'], subset[metric], label=f'{dist_metric}')
    plt.xlabel('k')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} vs k for different distance metrics')
    plt.legend()
    plt.show()
