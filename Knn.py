import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

df = pd.read_csv('C:/Users/PC/Desktop/New folder/slides/semester 10 - spring/cs488 AI/assing/Algorithms/Iris.csv')

X = df.drop('Species', axis=1)  
y = df['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
#X_train, X_test, y_train, y_test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(X, y))

def manhattan_distance(rec1, rec2):
    dist = 0
    for i in range(1,len(rec1)):
        dist += abs(rec1[i] - rec2[i])
    return dist

def euclidean_distance(rec1, rec2):
    dist = 0
    for i in range(1,len(rec1)-1):
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
        
    
        
        
# print(knn(X_train,y_train,X_train.loc[49],5,"euc"))
# print(knn(X_train,y_train,X_train.iloc[X_train['Id'] == 50],5))
#print(X_train.loc[49]) to see the id by train


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

    return [accuracy, precision, recall, f1_score_value]


""" def display_performance(X_train, y_train, X_test, y_test):
    performance_metrics = ['K Value',"Distance Metrics","Accuracy","Precision","Recall","F1 Score"]
    print(calculate_performance(X_train, y_train, X_test, y_test, 1, "man"))
    for i in range (4):
        performance_metrics.append([i,"Manhattan Distance"]+calculate_performance(X_train, y_train, X_test, y_test, i, "man"))
        performance_metrics.append([i,"Euclidean Distance"]+calculate_performance(X_train, y_train, X_test, y_test, i, "euc"))

    for record in performance_metrics:
        print(record) """
        

def display_performance(X_train, y_train, X_test, y_test):
    performance_metrics = pd.DataFrame(columns=['K Value', 'Distance Metrics', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    for i in range(1,10):
        performance_metrics.loc[len(performance_metrics)] = [i, "Manhattan Distance"] + calculate_performance(X_train, y_train, X_test, y_test, i, "man")
        performance_metrics.loc[len(performance_metrics)] = [i, "Euclidean Distance"] + calculate_performance(X_train, y_train, X_test, y_test, i, "euc")

    print("Performance Metrics:")
    print(performance_metrics)


def plot_decision_boundaries(X_train,y_train,X, y, h=0.02):
    y_pred = []
    for record in X.to_numpy():
        print(record)
        y_pred.append(knn(X_train,y_train,record,5,"man"))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.figure()
    
    plt.contourf(xx, yy, y_pred, cmap=cmap_light)
    # plt.pcolormesh(xx, yy, y_pred, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"3-Class classification (k = {clf.n_neighbors})")
    plt.show()

# plot_decision_boundaries(X_train,y_train,X_test.iloc[:2],y_test)

display_performance(X_train,y_train,X_test,y_test)