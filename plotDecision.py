import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Load the dataset
iris = load_iris()
X = iris.data[:, :2]  # We only take the first two features for visualization
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


def plot_decision_boundaries00(X, y, k):
    h = .02  # Step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Create color maps
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"3-Class classification (k = {k})")
    plt.show()

# Visualize decision boundaries for different k values
# for k in [1, 5, 10, 20]:
    # plot_decision_boundaries(X_train, y_train, k)

k_values = range(1, 21)
accuracy_scores = []
'''
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('KNN Accuracy for Different k Values')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

distance_metrics = ['euclidean', 'manhattan']
for metric in distance_metrics:
    accuracy_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    plt.plot(k_values, accuracy_scores, marker='o', label=metric)
plt.title('KNN Accuracy for Different Distance Metrics')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''


def plot_decision_boundaries(X_train, y_train, X_test, y_test, ks,metric):
    iris = load_iris()
    X = iris.data[:, :2]  # We only take the first two features for visualization
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    h = .02  # Step size in the mesh
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Create color maps
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    axs = axs.ravel()

    for i, k in enumerate(ks):
        # Plot the decision boundary
        clf = KNeighborsClassifier(n_neighbors=k,metric=metric)
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axs[i].pcolormesh(xx, yy, Z, cmap=cmap_light)
        axs[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
        axs[i].set_xlim(xx.min(), xx.max())
        axs[i].set_ylim(yy.min(), yy.max())
        axs[i].set_title(f"3-Class classification (k = {k})")

        # Calculate and display performance metrics
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # axs[i].text(0.05, 0.95, f"Accuracy: {accuracy:.2f}\nF1: {f1:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}", 
        #             transform=axs[i].transAxes, va='top', fontsize=8)

    plt.show()

# Example usage
plot_decision_boundaries(X_train, y_train, X_test, y_test, [1, 5, 10, 20])