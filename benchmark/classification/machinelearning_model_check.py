import joblib

model_path = "./benchmark/results/machinelearning_models/SVC_0.9290.pkl"

model = joblib.load(model_path)
print(model['parameters'])
