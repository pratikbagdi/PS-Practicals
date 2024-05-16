import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the kNN classifier
k = 3  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print correct and incorrect predictions
correct_predictions = []
incorrect_predictions = []
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_predictions.append((X_test[i], y_test[i], y_pred[i]))
    else:
        incorrect_predictions.append((X_test[i], y_test[i], y_pred[i]))

print("\nCorrect predictions:")
for item in correct_predictions:
    print("Input:", item[0], "| True Label:", iris.target_names[item[1]], "| Predicted Label:", iris.target_names[item[2]])

print("\nIncorrect predictions:")
for item in incorrect_predictions:
    print("Input:", item[0], "| True Label:", iris.target_names[item[1]], "| Predicted Label:", iris.target_names[item[2]])
