import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd

# Load the data
X_full = np.load(r"C:\Users\hiran\Desktop\SEM7\DAL\Final_Challenge\test_data.npy")
print("Loaded Data...", X_full.shape)

pca = joblib.load(r"C:\Users\hiran\Desktop\SEM7\DAL\Final_Challenge\pca.pkl")
X_reduced = pca.transform(X_full)

# Load the trained model and make predictions
model = joblib.load(r"C:\Users\hiran\Desktop\SEM7\DAL\Final_Challenge\LR_19.pkl")
y_binary = model.predict(X_reduced)
y_binary = (y_binary > 0.5).astype(int)
y_pred = y_binary

print("Predictions done...Working on final conversion")

# Load the MultiLabelBinarizer and use inverse_transform
mlb_new = joblib.load(r"C:\Users\hiran\Desktop\SEM7\DAL\Final_Challenge\mlb_new.pkl")
y_pred = mlb_new.inverse_transform(y_binary)  

print("Writing output file", len(y_pred))

submission_df = pd.DataFrame({
    "id": range(1, len(y_pred) + 1),  # Generates an id column starting from 1
    "labels": [";".join(sorted(labels)) if len(labels) > 0 else "" for labels in y_pred]  
})

# Save the DataFrame to a CSV file
submission_df.to_csv("submission_LR_19.csv", index=False)