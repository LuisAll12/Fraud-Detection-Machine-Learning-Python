import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib  # Speichern

# Lade den Datensatz
df = pd.read_csv('creditcard.csv')

# Ersten 5 Zeilen
print(df.head())

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Trainiere das Modell
model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für die ROC-AUC-Bewertung


# Klassifikationsbericht
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC-Score
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Speichere das Modell
joblib.dump(model, 'fraud_detection_model.pkl')

# Speichere den Scaler (falls du neue Daten skalieren möchtest)
joblib.dump(scaler, 'scaler.pkl')

from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Berechne Precision, Recall und ROC-AUC
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)


# Lade das Modell und den Scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Beispielhafte Transaktionsdaten (ersetze dies mit den Daten aus deinem UI)
new_data = {
    'Time': [150000],  # Ungewöhnliche Zeit (z. B. nachts)
    'V1': [3.0],       # Extrem hoher Wert
    'V2': [-2.5],      # Extrem niedriger Wert
    'V3': [2.8],       # Extrem hoher Wert
    'V4': [-3.0],      # Extrem niedriger Wert
    'V5': [1.5],       # Ungewöhnlicher Wert
    'V6': [-1.8],      # Ungewöhnlicher Wert
    'V7': [2.0],       # Ungewöhnlicher Wert
    'V8': [-2.2],      # Ungewöhnlicher Wert
    'V9': [1.7],       # Ungewöhnlicher Wert
    'V10': [-1.9],     # Ungewöhnlicher Wert
    'V11': [2.3],      # Ungewöhnlicher Wert
    'V12': [-2.1],     # Ungewöhnlicher Wert
    'V13': [1.8],      # Ungewöhnlicher Wert
    'V14': [-2.0],     # Ungewöhnlicher Wert
    'V15': [1.9],      # Ungewöhnlicher Wert
    'V16': [-1.7],     # Ungewöhnlicher Wert
    'V17': [2.1],      # Ungewöhnlicher Wert
    'V18': [-1.6],     # Ungewöhnlicher Wert
    'V19': [1.5],      # Ungewöhnlicher Wert
    'V20': [-1.4],     # Ungewöhnlicher Wert
    'V21': [1.3],      # Ungewöhnlicher Wert
    'V22': [-1.2],     # Ungewöhnlicher Wert
    'V23': [1.1],      # Ungewöhnlicher Wert
    'V24': [-1.0],     # Ungewöhnlicher Wert
    'V25': [0.9],      # Ungewöhnlicher Wert
    'V26': [-0.8],     # Ungewöhnlicher Wert
    'V27': [0.7],      # Ungewöhnlicher Wert
    'V28': [-0.6],     # Ungewöhnlicher Wert
    'Amount': [5000.0]  # Sehr hoher Betrag
}

# Konvertiere die Daten in ein DataFrame
new_df = pd.DataFrame(new_data)

# Skaliere den 'Amount'-Wert
new_df['Amount'] = scaler.transform(new_df[['Amount']])

# Mache eine Vorhersage
prediction = model.predict(new_df)
prediction_proba = model.predict_proba(new_df)[:, 1]

# Finde einen Betrugsfall im Testdatensatz
# fraud_sample = X_test[y_test == 1].iloc[0:1]

# Mache eine Vorhersage für diesen Fall
prediction = model.predict(new_df)
prediction_proba = model.predict_proba(new_df)[:, 1]

print("Vorhersage:", "Betrug" if prediction[0] == 1 else "Kein Betrug")
print("Wahrscheinlichkeit für Betrug:", prediction_proba[0])