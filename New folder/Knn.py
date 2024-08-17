import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap




df = pd.read_csv('./Iris.csv')

X = df.drop('Species', axis=1)  
y = df['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

def manhattan_distance(rec1, rec2):
    dist = 0
    iter = min(len(rec1),len(rec2))
    for i in range(1,iter):
        dist += abs(rec1[i] - rec2[i])
    return dist

def euclidean_distance(rec1, rec2):
    dist = 0
    iter = min(len(rec1),len(rec2))
    
    for i in range(1,(iter)):
        dist += (rec1[i] - rec2[i])**2
    return dist ** 0.5
def knn(training_set,training_class, new_record,k,distance_metrics):
  
    ts_array= training_set.to_numpy()
    distances = [[0 for _ in range(2)] for _ in range(len(ts_array))]   
    for i in range(len(ts_array)):
        distances[i][0] = training_class.iloc[i]
        if(distance_metrics == "man"):
            distances[i][1] =manhattan_distance(ts_array[i],new_record)
        elif(distance_metrics == "euc"):
            distances[i][1] =euclidean_distance(ts_array[i],new_record)
    sorted_distances = sorted(distances, key=lambda x: x[1])[:k]

    Iris_setosa = 0    
    Iris_virginica = 0 
    Iris_versicolor = 0

    for val in sorted_distances:
        if(val[0] == "Iris-setosa"):
            Iris_setosa += 1
        elif (val[0] == "Iris-virginica"):
            Iris_virginica += 1
        elif (val[0] == "Iris-versicolor"):
            Iris_versicolor+=1
    if((Iris_setosa+Iris_virginica+Iris_versicolor)!= len(sorted_distances)):
        print(Iris_setosa+Iris_virginica+Iris_versicolor," ",len(sorted_distances))
        print("ERRORR it does not add up!")
        
    max_value = max(Iris_setosa,Iris_versicolor,Iris_virginica)
    if(max_value == Iris_setosa):
        return "Iris-setosa"
    elif(max_value == Iris_versicolor):
        return "Iris-versicolor"
    elif(max_value == Iris_virginica):
        return "Iris-virginica"

def calculate_performance(X_train, y_train, X_test, y_test, k, distance_metric):
    correct_predictions = 0
    total_predictions = len(X_test)
    y_pred = []
    y_test_array = y_test.to_numpy()
    for i, test_row in enumerate(X_test.iterrows()):
        _, test_row = test_row
        predicted_label = knn(X_train, y_train, test_row, k, distance_metric)
        y_pred.append(predicted_label)
        if predicted_label == y_test.iloc[i]:
            correct_predictions += 1

    accuracy = accuracy_score(y_test_array, y_pred) #correct_predictions / total_predictions
    precision = precision_score(y_test_array, y_pred,average='macro') #precision(y_test_array, y_pred)
    recall = recall_score(y_test_array, y_pred,average='macro') #recall(y_test_array, y_pred)
    f1_score_value = f1_score(y_test_array, y_pred,average='macro') #f1_score(y_test_array, y_pred)

    return accuracy, precision, recall, f1_score_value


        

def display_performance(X_train, y_train, X_test, y_test):
    performance_metrics = pd.DataFrame(columns=['K Value', 'Distance Metrics', 'Accuracy %', 'Precision %', 'Recall', 'F1 Score'])

    for i in range(1,10):
        accuracy, precision, recall,f1 = calculate_performance(X_train, y_train, X_test, y_test, i, "man")
        performance_metrics.loc[len(performance_metrics)] = [i, "Manhattan Distance"] + [(accuracy*100),(precision*100),recall,f1]
        accuracy1, precision1, recall1,f11 = calculate_performance(X_train, y_train, X_test, y_test, i, "euc")
        performance_metrics.loc[len(performance_metrics)] = [i, "Euclidean Distance"] + [(accuracy1*100),(precision1*100),recall1,f11]

    print("Performance Metrics:")
    print(performance_metrics)


def decision_boundary(X_train, y_train, k, distance_metric):

    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = np.array([knn(X_train, y_train, np.array([x, y]), k, distance_metric) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    cmap = ListedColormap(['r', 'g', 'b'])
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cmap, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary (k={k}, distance metric={distance_metric})')
    plt.show()

def plot_performance_metrics(X_train, y_train, X_test, y_test):
    performance_metrics = pd.DataFrame(columns=['K Value', 'Distance Metrics', 'Accuracy %', 'Precision %', 'Recall', 'F1 Score'])

    for i in range(1, 10):
        accuracy, precision, recall, f1 = calculate_performance(X_train, y_train, X_test, y_test, i, "man")
        performance_metrics.loc[len(performance_metrics)] = [i, "Manhattan Distance"] + [(accuracy*100), (precision*100), recall, f1]
        accuracy, precision, recall, f1 = calculate_performance(X_train, y_train, X_test, y_test, i, "euc")
        performance_metrics.loc[len(performance_metrics)] = [i, "Euclidean Distance"] + [(accuracy*100), (precision*100), recall, f1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(performance_metrics['K Value'], performance_metrics['Accuracy %'], marker='o', label='Manhattan Distance')
    axes[0, 0].plot(performance_metrics['K Value'], performance_metrics['Accuracy %'], marker='x', label='Euclidean Distance')
    axes[0, 0].set_title('Accuracy vs K-Value')
    axes[0, 0].set_xlabel('K-Value')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()

    axes[0, 1].plot(performance_metrics['K Value'], performance_metrics['Precision %'], marker='o', label='Manhattan Distance')
    axes[0, 1].plot(performance_metrics['K Value'], performance_metrics['Precision %'], marker='x', label='Euclidean Distance')
    axes[0, 1].set_title('Precision vs K-Value')
    axes[0, 1].set_xlabel('K-Value')
    axes[0, 1].set_ylabel('Precision (%)')
    axes[0, 1].legend()

    axes[1, 0].plot(performance_metrics['K Value'], performance_metrics['Recall'], marker='o', label='Manhattan Distance')
    axes[1, 0].plot(performance_metrics['K Value'], performance_metrics['Recall'], marker='x', label='Euclidean Distance')
    axes[1, 0].set_title('Recall vs K-Value')
    axes[1, 0].set_xlabel('K-Value')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()

    axes[1, 1].plot(performance_metrics['K Value'], performance_metrics['F1 Score'], marker='o', label='Manhattan Distance')
    axes[1, 1].plot(performance_metrics['K Value'], performance_metrics['F1 Score'], marker='x', label='Euclidean Distance')
    axes[1, 1].set_title('F1-Score vs K-Value')
    axes[1, 1].set_xlabel('K-Value')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return performance_metrics


def plot_decision_boundaries(metric):
    iris = load_iris()
    X = iris.data[:, :2]  
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    h = .02  # Step size in the mesh
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    axs = axs.ravel()

    for i, k in enumerate([1, 5, 10, 20]):
        clf = KNeighborsClassifier(n_neighbors=k,metric=metric)
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axs[i].pcolormesh(xx, yy, Z, cmap=cmap_light)
        axs[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
        axs[i].set_xlim(xx.min(), xx.max())
        axs[i].set_ylim(yy.min(), yy.max())
        axs[i].set_title(f"3-Class classification (k = {k})")

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
    plt.show()



# Call the function to plot


# plot_knn_metrics()
plot_performance_metrics(X_train,y_train,X_test,y_test)
display_performance(X_train,y_train,X_test,y_test)
plot_decision_boundaries( "manhattan")
plot_decision_boundaries( "euclidean")