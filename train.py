import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
data = pd.read_csv('../creditcard.csv')  # Replace with your dataset path

# Separate features (X) and target (y)
X = data.drop('Class', axis=1)  # Features (all columns except 'Class')
y = data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (e.g., RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")