"""
phishing_detection.py
----------------------
END-TO-END PIPELINE for Phishing Website Detection using Machine Learning.

Pipeline stages (each clearly marked with a header comment, matching the
structure you'll present in your viva):
    1. Problem Understanding   (see README.md / docstring below)
    2. Dataset Loading
    3. Exploratory Data Analysis (EDA)
    4. Visualization
    5. Preprocessing (encoding + scaling)
    6. Train-Test Split
    7. Model Training (Logistic Regression + Random Forest)
    8. Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
    9. Model Comparison
   10. Prediction on New Input
   11. Saving the model + scaler for the Streamlit app (app.py)

Run with:  python phishing_detection.py
All plots are saved into the "plots/" folder (so they still work on headless
servers where a GUI window can't pop up).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures to disk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

os.makedirs("plots", exist_ok=True)
sns.set_style("whitegrid")

# =====================================================================
# STEP 1: PROBLEM UNDERSTANDING  (for viva explanation, kept as a comment)
# =====================================================================
"""
Phishing websites impersonate legitimate sites (banks, email providers,
e-commerce portals) to trick users into revealing sensitive information
such as passwords, OTPs, or card numbers.

Traditional defense = BLACKLISTING known phishing URLs.
Problem with blacklists:
    - Reactive, not proactive: a URL must already be reported/verified as
      phishing before it gets blacklisted -- new (zero-hour) phishing sites
      slip through easily.
    - Attackers rotate domains/URLs constantly, so blacklists go stale fast.
    - No way to generalize: blacklists can't say "this NEW, never-seen URL
      *looks* like phishing based on its structure."

Machine Learning approach:
    - Extracts FEATURES from a URL/webpage (URL length, use of IP address,
      HTTPS validity, domain age, use of '@', subdomains, etc.).
    - Learns the PATTERN that separates phishing from legitimate sites from
      historical labelled data.
    - Can classify a completely new, never-before-seen URL instantly,
      making it proactive rather than reactive.
    - This is framed as a SUPERVISED BINARY CLASSIFICATION problem:
      Result = 1 (Phishing) or Result = 0 (Legitimate).
"""

# =====================================================================
# STEP 2: DATASET LOADING
# =====================================================================
print("=" * 70)
print("STEP 2: LOADING DATASET")
print("=" * 70)

df = pd.read_csv("phishing_dataset.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}")
print(df.head())

# =====================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\n--- Shape ---")
print(df.shape)

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Null values per column ---")
print(df.isnull().sum())

print(f"\n--- Duplicate rows ---\n{df.duplicated().sum()} duplicate rows found")

print("\n--- Statistical summary (describe) ---")
print(df.describe())

# ---- Handle missing values ----
# Numeric NaNs -> fill with column median (robust to outliers, keeps
# distribution shape better than mean for skewed features).
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
print("\nMissing values handled using median imputation.")

# ---- Handle duplicates ----
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - df.shape[0]} duplicate rows. New shape: {df.shape}")

# =====================================================================
# STEP 4: VISUALIZATION
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: VISUALIZATION  (saved to ./plots/)")
print("=" * 70)

# 4.1 Count Plot - target class balance
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Result", data=df, hue="Result", palette=["#2ecc71", "#e74c3c"], legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Legitimate (0)", "Phishing (1)"])
plt.title("Class Balance: Phishing vs Legitimate")
plt.ylabel("Count")
plt.xlabel("")
plt.tight_layout()
plt.savefig("plots/1_countplot_class_balance.png", dpi=120)
plt.close()

# 4.2 Histogram - distribution of URL_Length
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="URL_Length", hue="Result", bins=30, kde=True,
             palette=["#2ecc71", "#e74c3c"])
plt.title("Histogram: URL Length Distribution (by class)")
plt.tight_layout()
plt.savefig("plots/2_histogram_url_length.png", dpi=120)
plt.close()

# 4.3 Scatter Plot - two continuous features
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Domain_Age_Days", y="TTL_Hostname",
                 hue="Result", palette=["#2ecc71", "#e74c3c"], alpha=0.6)
plt.title("Scatter: Domain Age vs Hostname TTL")
plt.tight_layout()
plt.savefig("plots/3_scatter_age_vs_anchor.png", dpi=120)
plt.close()

# 4.4 Bar Plot - average SSL/HTTPS validity per class
plt.figure(figsize=(6, 4))
sns.barplot(x="Result", y="Has_Valid_SSL", data=df, hue="Result",
            palette=["#2ecc71", "#e74c3c"], legend=False)
plt.xticks([0, 1], ["Legitimate", "Phishing"])
plt.title("Bar Plot: Avg. Valid HTTPS Rate by Class")
plt.tight_layout()
plt.savefig("plots/4_barplot_ssl_by_class.png", dpi=120)
plt.close()

# 4.5 Correlation Heatmap
plt.figure(figsize=(12, 9))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig("plots/5_correlation_heatmap.png", dpi=120)
plt.close()

print("Saved 5 plots: countplot, histogram, scatter, barplot, heatmap.")

# =====================================================================
# STEP 5: PREPROCESSING
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: PREPROCESSING")
print("=" * 70)

# All our features are already numeric (0/1 flags or numeric measurements),
# so no Label/One-Hot Encoding is needed here. In a real scraped dataset you
# might have a categorical column like "domain_registrar" -- example shown
# below (commented) for viva reference:
#
#   from sklearn.preprocessing import LabelEncoder
#   le = LabelEncoder()
#   df['registrar_encoded'] = le.fit_transform(df['domain_registrar'])

FEATURES = [c for c in df.columns if c != "Result"]
TARGET = "Result"

X = df[FEATURES]
y = df[TARGET]

# Feature Scaling: Logistic Regression is distance/gradient based, so
# features on very different scales (e.g. URL_Length ~0-200 vs flags 0/1)
# must be standardized to mean=0, std=1 for stable, fast convergence.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

print("Feature scaling applied using StandardScaler.")

# =====================================================================
# STEP 6: TRAIN-TEST SPLIT
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# =====================================================================
# STEP 7: MODEL TRAINING
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: MODEL TRAINING")
print("=" * 70)

# Primary model: Logistic Regression (simple, interpretable, fast baseline
# for binary classification -- gives probability of phishing directly).
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
print("Logistic Regression trained.")

# Comparison model: Random Forest (ensemble of decision trees, usually
# captures non-linear feature interactions better -> often higher accuracy).
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# =====================================================================
# STEP 8: EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("STEP 8: EVALUATION")
print("=" * 70)


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"plots/6_confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=120)
    plt.close()

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1}


results = []
results.append(evaluate_model(log_reg, X_test, y_test, "Logistic Regression"))
results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

# =====================================================================
# STEP 9: MODEL COMPARISON
# =====================================================================
print("\n" + "=" * 70)
print("STEP 9: MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
print(f"\nBest performing model (by F1-Score): {best_model_name}")

# Bar chart comparing both models across metrics
metrics_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(8, 5))
sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="Set2")
plt.title("Model Comparison: Logistic Regression vs Random Forest")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("plots/7_model_comparison.png", dpi=120)
plt.close()

# =====================================================================
# STEP 10: PREDICTION ON NEW INPUT
# =====================================================================
print("\n" + "=" * 70)
print("STEP 10: PREDICTION ON NEW (SAMPLE) INPUT")
print("=" * 70)

# Choose the best model going forward (Random Forest is used in the app,
# but you can switch to log_reg if you want to emphasize LR as primary).
final_model = rf if best_model_name == "Random Forest" else log_reg


def predict_new_sample(feature_dict, model=final_model, scaler=scaler, feature_order=FEATURES):
    """
    Takes a dictionary of raw feature values (same keys as FEATURES),
    scales them with the SAME scaler used in training, and returns:
        - predicted label: "Phishing" or "Legitimate"
        - probability of being phishing
    """
    input_df = pd.DataFrame([feature_dict], columns=feature_order)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_order)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # probability of class 1 (phishing)
    label = "Phishing" if pred == 1 else "Legitimate"
    return label, proba


# Example: a suspicious-looking URL's extracted features
sample_input = {
    "URL_Length": 120,
    "Domain_Length": 35,
    "Having_At_Symbol": 1,
    "Having_Hyphen_In_Domain": 1,
    "Num_Dots_In_Domain": 3,
    "Is_Shortened_URL": 1,
    "Num_Slashes_In_URL": 6,
    "Has_Email_In_URL": 1,
    "Has_Valid_SSL": 0,
    "Has_SPF_Record": 0,
    "Has_DNS_Nameservers": 0,
    "Num_Mail_Servers": 0,
    "Domain_Age_Days": 15,
    "Domain_Expiry_Days": 30,
    "TTL_Hostname": 300,
    "Num_Redirects": 5,
    "Server_Response_Time": 2.5,
    "Num_IPs_Resolved": 1,
}

label, proba = predict_new_sample(sample_input)
print(f"Sample input prediction -> {label}  (Phishing probability: {proba:.2%})")

# =====================================================================
# STEP 11: SAVE ARTIFACTS FOR STREAMLIT APP
# =====================================================================
print("\n" + "=" * 70)
print("STEP 11: SAVING MODEL + SCALER FOR THE WEB APP")
print("=" * 70)

joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "feature_names.pkl")

print("Saved: logistic_regression_model.pkl, random_forest_model.pkl, scaler.pkl, feature_names.pkl")
print("\nPipeline complete. Ready to run the Streamlit app: streamlit run app.py")
