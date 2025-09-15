import pandas as pd
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model("models/xgb_model.json")

# Load new data
new_df = pd.read_csv("data/new_transactions.csv")

# Keep IDs for reporting
transaction_ids = new_df["nameOrig"]

# Drop non-numeric columns for prediction
X_new = new_df.drop(columns=["nameOrig", "nameDest", "type"])

# Predict probabilities
fraud_prob = model.predict_proba(X_new)[:, 1]

# Add results back
new_df["fraud_probability"] = fraud_prob
new_df["prediction"] = (fraud_prob >= 0.5).astype(int)

# Split into groups
frauds_high_conf = new_df[(new_df["prediction"] == 1) & (new_df["fraud_probability"] >= 0.9)]
not_frauds_high_conf = new_df[(new_df["prediction"] == 0) & (new_df["fraud_probability"] >= 0.9)]
uncertain = new_df[new_df["fraud_probability"] < 0.9]

# Print results
print("\n=== Predicted Fraud (>90% confidence) ===")
print(frauds_high_conf[["nameOrig", "fraud_probability"]])

print("\n=== Predicted Not Fraud (>90% confidence) ===")
print(not_frauds_high_conf[["nameOrig", "fraud_probability"]])

print("\n=== Uncertain (<90% confidence, needs review) ===")
print(uncertain[["nameOrig", "fraud_probability"]])

# Save predictions
new_df.to_csv("data/new_predictions.csv", index=False)
print("\nPredictions saved to data/new_predictions.csv")
