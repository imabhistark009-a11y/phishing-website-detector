

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------------
# LOAD SAVED MODEL, SCALER, FEATURE NAMES  (cached so it loads only once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("random_forest_model.pkl")
    lr_model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return rf_model, lr_model, scaler, feature_names


rf_model, lr_model, scaler, FEATURES = load_artifacts()

# ---------------------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------------------
st.markdown("""
<style>
    .main { padding-top: 1rem; }

    /* Hero header banner */
    .hero {
        background: linear-gradient(135deg, #1f2a44 0%, #2c3e50 100%);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 { color: #ffffff !important; margin: 0 0 0.3rem 0; font-size: 1.9rem; }
    .hero p { color: #b8c4d4 !important; margin: 0; font-size: 0.95rem; }

    /* Section labels */
    .section-label {
        font-weight: 700; font-size: 1.05rem; margin: 1.2rem 0 0.4rem 0;
        color: inherit;
    }

    /* Result cards — text color fixed explicitly so it stays readable
       regardless of Streamlit's light/dark theme */
    .result-safe {
        background-color: #e8f8ee; border-left: 6px solid #2ecc71;
        padding: 1.1rem 1.4rem; border-radius: 10px; margin-top: 1rem;
        color: #14532d !important;
    }
    .result-safe h3, .result-safe p { color: #14532d !important; margin: 0.2rem 0; }

    .result-danger {
        background-color: #fdecea; border-left: 6px solid #e74c3c;
        padding: 1.1rem 1.4rem; border-radius: 10px; margin-top: 1rem;
        color: #7f1d1d !important;
    }
    .result-danger h3, .result-danger p { color: #7f1d1d !important; margin: 0.2rem 0; }

    /* Probability gauge bar */
    .gauge-track {
        background-color: rgba(120,120,120,0.25); border-radius: 20px;
        height: 14px; width: 100%; margin: 0.6rem 0 0.2rem 0; overflow: hidden;
    }
    .gauge-fill {
        height: 100%; border-radius: 20px;
        transition: width 0.4s ease-in-out;
    }

    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em;
        font-weight: 600; background-color: #2c3e50; color: white;
        border: none;
    }
    .stButton>button:hover { background-color: #34495e; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛡️ Phishing Website Detector</h1>
    <p>Machine Learning based classifier — enter website characteristics below and get an instant Phishing / Legitimate prediction.</p>
</div>
""", unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Choose model",
    ["Random Forest", "Logistic Regression"],
    help="Both were trained on the same dataset; compare their predictions."
)
model = rf_model if model_choice == "Random Forest" else lr_model

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.info(
    "This tool predicts whether a website is likely **Phishing** or "
    "**Legitimate** using real URL, domain-trust, and DNS-based features "
    "(sourced from a published academic dataset), trained with Logistic "
    "Regression / Random Forest classifiers."
)

# ---------------------------------------------------------------------
# QUICK HELPER: auto-fill a couple of fields from a pasted URL string
# ---------------------------------------------------------------------
st.markdown('<div class="section-label">1️⃣ Quick URL scan (optional)</div>', unsafe_allow_html=True)
url_input = st.text_input(
    "Paste a URL here to auto-fill some fields below",
    placeholder="e.g. http://192.168.1.1/secure-login@paypal.com"
)

auto_url_length = len(url_input) if url_input else 40
auto_has_at = 1 if "@" in (url_input or "") else 0
auto_has_hyphen = 1 if "-" in (url_input.split("//")[-1].split("/")[0] if url_input else "") else 0
auto_has_https = 1 if (url_input or "").lower().startswith("https") else 0
auto_domain_length = len(url_input.split("//")[-1].split("/")[0]) if url_input else 15
auto_num_slashes = url_input.count("/") if url_input else 3

if url_input:
    st.caption(
        f"Auto-detected → URL Length: {auto_url_length} | "
        f"Domain Length: {auto_domain_length} | "
        f"Has '@' symbol: {'Yes' if auto_has_at else 'No'} | "
        f"HTTPS: {'Yes' if auto_has_https else 'No'} | "
        f"Has '-' in domain: {'Yes' if auto_has_hyphen else 'No'}"
    )

st.markdown('<div class="section-label">2️⃣ Website / URL characteristics</div>', unsafe_allow_html=True)

with st.expander("🔗 URL & domain structure", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        url_length = st.slider("URL Length (characters)", 5, 250, int(auto_url_length))
        domain_length = st.slider("Domain Length (characters)", 3, 60, int(auto_domain_length))
        having_at = st.selectbox("Contains '@' symbol in URL?",
                                  ["No", "Yes"], index=auto_has_at)
        having_hyphen = st.selectbox("Domain contains '-' (e.g. paypal-secure.com)?",
                                      ["No", "Yes"], index=auto_has_hyphen)
    with col2:
        num_dots_domain = st.slider("Number of dots '.' in domain (subdomain proxy)", 0, 5, 1)
        is_shortened = st.selectbox("Uses a URL shortener (bit.ly, tinyurl, etc.)?",
                                     ["No", "Yes"], index=0)
        num_slashes = st.slider("Number of '/' in the URL", 0, 20, int(auto_num_slashes))
        has_email = st.selectbox("Contains an email address in the URL?",
                                  ["No", "Yes"], index=0)

with st.expander("🔒 Certificate & DNS trust signals", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        https_valid = st.selectbox("Valid HTTPS with trusted certificate?",
                                    ["No", "Yes"], index=auto_has_https)
        has_spf = st.selectbox("Domain has a valid SPF (anti-spoofing email) record?",
                                ["No", "Yes"], index=1)
    with col4:
        has_dns = st.selectbox("Domain has valid DNS nameservers?", ["No", "Yes"], index=1)
        num_mx = st.slider("Number of mail servers (MX records)", 0, 10, 2)

with st.expander("🌐 Domain age & network behaviour", expanded=True):
    col5, col6 = st.columns(2)
    with col5:
        domain_age_days = st.slider("Domain age (days since registration)", 0, 8000, 365)
        domain_expiry_days = st.slider("Days until domain registration expires", 0, 3000, 200)
        ttl_hostname = st.slider("Hostname DNS TTL (seconds)", 0, 70000, 3600)
    with col6:
        num_redirects = st.slider("Number of redirects observed", 0, 10, 1)
        response_time = st.slider("Server response time (seconds)", 0.0, 10.0, 0.5)
        num_ips = st.slider("Number of IPs the domain resolves to", 0, 10, 1)

yn = {"No": 0, "Yes": 1}

# ---------------------------------------------------------------------
# BUILD FEATURE VECTOR IN THE EXACT ORDER THE MODEL WAS TRAINED ON
# ---------------------------------------------------------------------
input_dict = {
    "URL_Length": url_length,
    "Domain_Length": domain_length,
    "Having_At_Symbol": yn[having_at],
    "Having_Hyphen_In_Domain": yn[having_hyphen],
    "Num_Dots_In_Domain": num_dots_domain,
    "Is_Shortened_URL": yn[is_shortened],
    "Num_Slashes_In_URL": num_slashes,
    "Has_Email_In_URL": yn[has_email],
    "Has_Valid_SSL": yn[https_valid],
    "Has_SPF_Record": yn[has_spf],
    "Has_DNS_Nameservers": yn[has_dns],
    "Num_Mail_Servers": num_mx,
    "Domain_Age_Days": domain_age_days,
    "Domain_Expiry_Days": domain_expiry_days,
    "TTL_Hostname": ttl_hostname,
    "Num_Redirects": num_redirects,
    "Server_Response_Time": response_time,
    "Num_IPs_Resolved": num_ips,
}

st.markdown("---")

if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_dict], columns=FEATURES)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # P(phishing)

    gauge_pct = round(proba * 100, 1)
    # Gauge color scales from green (safe) to red (risky) based on probability
    gauge_color = "#e74c3c" if proba >= 0.5 else "#2ecc71"

    # NOTE: no leading indentation on these lines -- Markdown treats 4+ leading
    # spaces as a code block, which was causing stray closing tags (like
    # "</div>") to render as literal visible text instead of real HTML.
    gauge_html = (
        f'<div class="gauge-track">'
        f'<div class="gauge-fill" style="width:{gauge_pct}%; background-color:{gauge_color};"></div>'
        f'</div>'
    )

    if prediction == 1:
        st.markdown(
            f'<div class="result-danger">'
            f'<h3>⚠️ Likely PHISHING website</h3>'
            f'<p>Predicted phishing probability: <b>{proba:.1%}</b></p>'
            f'{gauge_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.warning("Do not enter personal information or credentials on this site.")
    else:
        st.markdown(
            f'<div class="result-safe">'
            f'<h3>✅ Likely LEGITIMATE website</h3>'
            f'<p>Predicted phishing probability: <b>{proba:.1%}</b></p>'
            f'{gauge_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.success("No strong phishing indicators detected.")

    with st.expander("See the exact feature values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # -----------------------------------------------------------------
    # RED-FLAG CHECKLIST — simple rule-based explanation, independent of
    # the model, so a viewer instantly sees *which* raw signals looked risky
    # -----------------------------------------------------------------
    st.markdown("#### 🚩 Red flags detected in this input")
    flags = [
        (input_dict["Having_At_Symbol"] == 1, "URL contains an '@' symbol"),
        (input_dict["Having_Hyphen_In_Domain"] == 1, "Domain contains a '-' (prefix/suffix trick)"),
        (input_dict["Num_Dots_In_Domain"] >= 3, "Domain has 3 or more dots (many subdomains)"),
        (input_dict["Is_Shortened_URL"] == 1, "URL uses a shortening service"),
        (input_dict["Has_Email_In_URL"] == 1, "URL contains an email address"),
        (input_dict["Has_Valid_SSL"] == 0, "No valid HTTPS / trusted certificate"),
        (input_dict["Has_SPF_Record"] == 0, "No SPF (anti-spoofing) record found"),
        (input_dict["Has_DNS_Nameservers"] == 0, "No valid DNS nameservers found"),
        (input_dict["Domain_Age_Days"] < 180, "Domain is less than 6 months old"),
        (input_dict["Num_Redirects"] >= 3, "URL has 3 or more redirects"),
    ]
    triggered = [msg for cond, msg in flags if cond]

    if triggered:
        for msg in triggered:
            st.markdown(f"- 🔴 {msg}")
    else:
        st.markdown("- 🟢 No common red flags triggered")

    # -----------------------------------------------------------------
    # FEATURE IMPORTANCE — shows which features the MODEL globally relies
    # on most, for interpretability during the viva
    # -----------------------------------------------------------------
    st.markdown("#### 📊 What the model weighs most heavily (overall)")
    if model_choice == "Random Forest":
        importances = pd.Series(model.feature_importances_, index=FEATURES)
    else:
        importances = pd.Series(np.abs(model.coef_[0]), index=FEATURES)

    top_importances = importances.sort_values(ascending=False).head(6)
    st.bar_chart(top_importances)

st.markdown("---")
st.caption(
    "Model trained on a labelled phishing/legitimate website dataset using "
    "Logistic Regression and Random Forest classifiers. For academic/"
    "demonstration purposes only — not a substitute for real-time security tools."
)
