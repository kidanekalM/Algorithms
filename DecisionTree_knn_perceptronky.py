import numpy as np
import pandas as pd  # Import pandas
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import Counter  # Import Counter

# Define a simple Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _split(self, X, y, index, threshold):
        left_mask = X[:, index] <= threshold
        right_mask = X[:, index] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        best_index, best_threshold = None, None
        best_gini = 1
        n_samples, n_features = X.shape

        for index in range(n_features):
            thresholds = np.unique(X[:, index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_index = index
                    best_threshold = threshold

        return best_index, best_threshold

    def _grow_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y[0]

        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        index, threshold = self._best_split(X, y)
        if index is None:
            return Counter(y).most_common(1)[0][0]

        X_left, X_right, y_left, y_right = self._split(X, y, index, threshold)
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)
        return (index, threshold, left_subtree, right_subtree)

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            index, threshold, left, right = node
            if inputs[index] <= threshold:
                node = left
            else:
                node = right
        return node

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, ax, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Sepal Length", fontsize=12)
    ax.set_ylabel("Sepal Width", fontsize=12)

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Classifier Comparison: Decision Tree, KNN, Perceptron", layout="wide", initial_sidebar_state="expanded")

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data[:, :2]  # Use only the first two features (Sepal Length and Sepal Width)
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sidebar for user input
    st.sidebar.header("🔧 Model Parameters")
    max_depth = st.sidebar.slider("Maximum Depth (for Decision Tree)", 1, 10, 5, 1)
    k_neighbors = st.sidebar.slider("Number of Neighbors (for KNN)", 1, 10, 5, 1)

    # Initialize models
    models = {
        "Decision Tree": DecisionTree(max_depth=max_depth),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=k_neighbors),
        "Perceptron": Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    }

    # Train models and predict
    predictions = {}
    metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[model_name] = y_pred
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Precision"].append(precision_score(y_test, y_pred, average='macro'))
        metrics["Recall"].append(recall_score(y_test, y_pred, average='macro'))
        metrics["F1-Score"].append(f1_score(y_test, y_pred, average='macro'))

    # # Display performance metrics
    # st.markdown("<h3>✨ Model Performance</h3>", unsafe_allow_html=True)
    # metric_df = pd.DataFrame(metrics, index=models.keys())

    # Display performance metrics with better visibility
    st.markdown("<h3>✨ Model Performance</h3>", unsafe_allow_html=True)
    metric_df = pd.DataFrame(metrics, index=models.keys())

    # Create one column to display all the metrics one by one
    st.bar_chart(metric_df, height=300)

    # Visualize decision boundaries
    st.markdown("<h3>🌐 Decision Boundaries</h3>", unsafe_allow_html=True)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (model_name, model) in zip(axs, models.items()):
        plot_decision_boundaries(X_train, y_train, model, ax, f"{model_name} (Train)")
    st.pyplot(fig)

    # Interactive Prediction
    st.markdown("<h3>🔍 Interactive Prediction</h3>", unsafe_allow_html=True)
    sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))

    input_data = np.array([[sepal_length, sepal_width]])
    st.write("### Prediction Results:")

    for model_name, model in models.items():
        prediction = model.predict(input_data)[0]
        st.write(f"{model_name} predicts this sample belongs to class: **{iris.target_names[prediction]}**")


if __name__ == "__main__":
    main()
