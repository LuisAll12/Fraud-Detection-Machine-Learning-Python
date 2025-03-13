from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load your model
with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # List of required fields
        fields = [
            'time', 'amount', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10',
            'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21',
            'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28'
        ]

        # Validate inputs
        input_data = []
        for field in fields:
            value = request.form.get(field, '').strip()  # Get the value and remove whitespace
            if not value:  # If the value is empty
                return render_template('index.html', error=f"Please fill in all fields. Missing: {field}")
            try:
                input_data.append(float(value))  # Convert to float
            except ValueError:
                return render_template('index.html', error=f"Invalid input for {field}. Please enter a number.")

        # Create a dictionary with the input data
        new_data = {
            'Time': [input_data[0]],  # Use the first value from input_data
            'V1': [input_data[2]],    # Use the third value from input_data
            'V2': [input_data[3]],    # Use the fourth value from input_data
            'V3': [input_data[4]],    # Continue for all fields...
            'V4': [input_data[5]],
            'V5': [input_data[6]],
            'V6': [input_data[7]],
            'V7': [input_data[8]],
            'V8': [input_data[9]],
            'V9': [input_data[10]],
            'V10': [input_data[11]],
            'V11': [input_data[12]],
            'V12': [input_data[13]],
            'V13': [input_data[14]],
            'V14': [input_data[15]],
            'V15': [input_data[16]],
            'V16': [input_data[17]],
            'V17': [input_data[18]],
            'V18': [input_data[19]],
            'V19': [input_data[20]],
            'V20': [input_data[21]],
            'V21': [input_data[22]],
            'V22': [input_data[23]],
            'V23': [input_data[24]],
            'V24': [input_data[25]],
            'V25': [input_data[26]],
            'V26': [input_data[27]],
            'V27': [input_data[28]],
            'V28': [input_data[29]],
            'Amount': [input_data[1]]  # Use the second value from input_data
        }
        fraud_data = {
            'Time': [150000],  # Unusual time (e.g., late at night)
            'V1': [-5.0],      # Very extreme negative value
            'V2': [4.5],       # Very extreme positive value
            'V3': [-6.0],      # Very extreme negative value
            'V4': [5.0],       # Very extreme positive value
            'V5': [-4.5],      # Very extreme negative value
            'V6': [4.0],       # Very extreme positive value
            'V7': [-5.5],      # Very extreme negative value
            'V8': [3.8],       # Very extreme positive value
            'V9': [-4.0],      # Very extreme negative value
            'V10': [4.2],      # Very extreme positive value
            'V11': [-5.2],     # Very extreme negative value
            'V12': [3.9],      # Very extreme positive value
            'V13': [-4.8],     # Very extreme negative value
            'V14': [4.1],      # Very extreme positive value
            'V15': [-5.1],     # Very extreme negative value
            'V16': [3.7],      # Very extreme positive value
            'V17': [-4.9],     # Very extreme negative value
            'V18': [4.3],      # Very extreme positive value
            'V19': [-5.3],     # Very extreme negative value
            'V20': [3.6],      # Very extreme positive value
            'V21': [-4.7],     # Very extreme negative value
            'V22': [4.4],      # Very extreme positive value
            'V23': [-5.4],     # Very extreme negative value
            'V24': [3.5],      # Very extreme positive value
            'V25': [-4.6],     # Very extreme negative value
            'V26': [4.0],      # Very extreme positive value
            'V27': [-5.0],     # Very extreme negative value
            'V28': [3.4],      # Very extreme positive value
            'Amount': [10000.0] # Very high amount
        }

        # Convert the dictionary to a DataFrame
        new_df = pd.DataFrame(fraud_data)

        # Make a prediction using the DataFrame
        prediction = model.predict(new_df)
        prediction_proba = model.predict_proba(new_df)[:, 1]

        print("Vorhersage:", "Betrug" if prediction[0] == 1 else "Kein Betrug")
        print("Wahrscheinlichkeit für Betrug:", prediction_proba[0])

        # Determine the result
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"

        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)