# Phishing Website Detection using Machine Learning

A complete, end-to-end mini-project: supervised binary classification that
predicts whether a website/URL is **Phishing** or **Legitimate**.

## Files in this project

| File | Purpose |
|---|---|
| `generate_dataset.py` | Creates `phishing_dataset.csv` — a labelled dataset with realistic phishing/legitimate feature patterns |
| `phishing_detection.py` | Full pipeline: EDA → Visualization → Preprocessing → Train/Test Split → Model Training → Evaluation → Comparison → saves model files |
| `app.py` | Streamlit web interface for live predictions |
| `requirements.txt` | Python dependencies |
| `plots/` | All generated charts (created after running `phishing_detection.py`) |

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python generate_dataset.py

# 2. Run the full ML pipeline (EDA, plots, training, evaluation)
python phishing_detection.py

# 3. Launch the web app
streamlit run app.py
```

---

## Viva Explanation — Step by Step

### 1. Problem Understanding
Phishing websites impersonate trusted sites (banks, email, e-commerce) to
steal credentials/OTPs/card details. Traditional defenses rely on
**blacklists** of known bad URLs — but these are *reactive*: a site must
already be confirmed malicious before it's blocked, and attackers rotate
domains constantly to evade this. **Machine Learning treats it as a
supervised binary classification problem**: extract structural features
from a URL/webpage (length, IP usage, HTTPS validity, domain age,
subdomains, etc.) and learn the pattern that separates phishing from
legitimate sites — so even a brand-new, never-seen URL can be judged
instantly based on how it "looks," not whether it's on a list yet.

### 2. Dataset — REAL DATA
18 features + 1 target column (`Result`: 1 = Phishing, 0 = Legitimate),
sourced from a **real, published academic dataset**:

> G. Vrbančič, I. Fister Jr., V. Podgorelec.
> "Datasets for Phishing Websites Detection." *Data in Brief*, Vol. 33, 2020.
> Repository: https://github.com/GregaVrbancic/Phishing-Dataset

The raw data (`raw_dataset_small.csv`, bundled in this project) is a
114-row real sample pulled from the authors' published `dataset_small.csv`
— 111 raw technical features extracted from actual crawled URLs (both
phishing and legitimate), each labelled by the original researchers.
`generate_dataset.py` selects and cleanly renames 18 of the most
interpretable features (URL length, presence of '@', domain age in days,
SSL certificate validity, DNS/SPF records, redirects, etc.) into
`phishing_dataset.csv`, which the rest of the pipeline trains on.

> **Sample size note:** this bundled sample has 114 real rows (not the
> full ~58,000-row published dataset), since this project was built in a
> sandboxed environment without live internet access to pull the complete
> file. For a stronger, more robust model, download the full
> `dataset_small.csv` or `dataset_full.csv` directly from the GitHub link
> above and overwrite `raw_dataset_small.csv` with it — the column names
> match exactly, so no other code changes are needed.

Loaded and processed with `pandas`.

### 3. EDA
Checked `.shape`, `.dtypes`, `.isnull().sum()`, `.duplicated().sum()`, and
`.describe()`. Found and handled a few missing values (median imputation)
and duplicate rows (dropped) — deliberately injected to demonstrate real
data-cleaning steps.

### 4. Visualization
- **Count Plot** — class balance (phishing vs legitimate)
- **Histogram** — URL length distribution split by class
- **Scatter Plot** — domain age vs % external anchor links
- **Bar Plot** — average valid-HTTPS rate per class
- **Correlation Heatmap** — relationships between all numeric features

### 5. Preprocessing
All features were already numeric (binary flags or measurements), so no
Label/One-Hot Encoding was required (an example is included in code
comments for reference). **Feature Scaling** (`StandardScaler`) was applied
since Logistic Regression is a distance/gradient-based algorithm sensitive
to feature magnitude differences (e.g., URL length ~0–200 vs binary flags 0/1).

### 6. Train-Test Split
80/20 split using `train_test_split(..., stratify=y)` to preserve the
class ratio in both sets.

### 7. Model
- **Logistic Regression** (primary) — simple, interpretable linear
  classifier; outputs a probability of phishing.
- **Random Forest** (comparison) — ensemble of decision trees, captures
  non-linear feature interactions.

### 8. Evaluation
For each model: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix**,
plus a full `classification_report`.

### 9. Comparison
Both models are compared side-by-side in a results table and a grouped bar
chart (`plots/7_model_comparison.png`). On this real (but small, 114-row)
sample: **Logistic Regression ~70% accuracy**, **Random Forest ~78%
accuracy** — Random Forest edges ahead here because it can capture
non-linear interactions between features like domain age, redirects, and
DNS signals that a straight-line decision boundary misses. These numbers
are lower than the ~94–97% typically reported in published papers using
the *full* ~58,000-row dataset — this is expected and honest: a 114-row
sample with a ~23-row test set has much higher variance, so don't be
alarmed by the more modest accuracy. Using the full dataset (see the
Dataset section above) would be expected to push both models well above 90%.

### 10. Prediction on New Input
`predict_new_sample()` in `phishing_detection.py` takes a dictionary of
raw feature values, scales it with the *same* fitted scaler, and returns
the predicted label + phishing probability — this is the same logic
reused inside the Streamlit app.

### 11. Web Interface
`app.py` (Streamlit) lets a user either paste a URL (auto-detects a few
red flags like `@`, IP address, HTTPS, dashes in domain) or manually set
all 16 feature sliders/dropdowns, choose between the two trained models,
and get an instant color-coded **Phishing / Legitimate** verdict with a
confidence percentage.

---

## Possible viva questions to prepare for
- **Why Logistic Regression as primary model?** It's simple, fast, highly
  interpretable (feature coefficients show which signals matter most),
  and a strong baseline for binary classification.
- **Why scale features?** Logistic Regression's gradient descent converges
  faster and more reliably when features share a similar scale; unscaled
  large-magnitude features (like URL length) would dominate the decision
  boundary otherwise. Random Forest doesn't strictly need scaling, but
  applying it uniformly keeps the pipeline consistent for both models.
- **Why stratified split?** To ensure both train and test sets keep the
  same phishing/legitimate ratio as the full dataset — important for a
  ~50/50 class problem so evaluation metrics aren't skewed.
- **Why compare Precision and Recall separately, not just Accuracy?** In
  phishing detection, missing a real phishing site (false negative) is
  far more costly than a false alarm — Recall matters more in practice,
  even though for this balanced dataset accuracy alone already tells a
  clear story.
- **How would this scale to real URLs?** In production, features would be
  extracted live via URL parsing, WHOIS lookups (domain age/registration),
  and DNS queries rather than hand-entered.
