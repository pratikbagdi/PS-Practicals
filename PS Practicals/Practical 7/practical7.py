# Install scikit-learn package if not installed
# pip install scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("climate_change_data.csv")

# Preprocessing
# We need to convert the 'Country' feature to numerical values using one-hot encoding
data = pd.get_dummies(data, columns=['Country'])
# Binning temperature into categories
temperature_bins = [-float('inf'), 10, 20, float('inf')]
temperature_labels = ['Low', 'Medium', 'High']
data['Temperature_Category'] = pd.cut(data['Temperature'], bins=temperature_bins, labels=temperature_labels)

# Define features (X) and target variable (y)
X = data.drop(['Date', 'Location', 'Temperature', 'Temperature_Category'], axis=1)
y = data['Temperature_Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose classifier (Random Forest Classifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate the performance of classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Other classification metrics
print(classification_report(y_test, predictions))
