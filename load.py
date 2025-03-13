import pickle
with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)