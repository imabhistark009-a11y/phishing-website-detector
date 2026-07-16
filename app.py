

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
# CUSTOM STYLING — "URL forensics scanner" visual identity
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --ink: #0A0E17;
    --panel: #121926;
    --line: #223047;
    --signal: #2DD4BF;
    --safe: #34D399;
    --danger: #F43F5E;
    --text: #E5EDF5;
    --muted: #8393AB;
}

.main { padding-top: 1rem; }
body, p, span, div, label { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* ---------- Hero header with animated scan-line ---------- */
.hero {
    background: linear-gradient(180deg, var(--panel) 0%, #0d1420 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.6rem 1.8rem 1.3rem 1.8rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 12px;
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid rgba(45, 212, 191, 0.35);
    font-size: 1.5rem; margin-bottom: 0.8rem;
}
.hero h1 {
    color: var(--text) !important; margin: 0 0 0.35rem 0;
    font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em;
}
.hero p { color: var(--muted) !important; margin: 0; font-size: 0.92rem; line-height: 1.5; }
.scan-line {
    position: absolute; bottom: 0; left: 0; height: 2px; width: 40%;
    background: linear-gradient(90deg, transparent, var(--signal), transparent);
    animation: sweep 3.2s ease-in-out infinite;
}
@keyframes sweep {
    0%   { left: -40%; }
    100% { left: 100%; }
}

/* ---------- Section headers ---------- */
.section-label {
    display: flex; align-items: center; gap: 0.55rem;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 1rem; letter-spacing: 0.01em;
    margin: 1.4rem 0 0.6rem 0; color: var(--text);
}
.section-label .badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; border-radius: 7px;
    background: rgba(45, 212, 191, 0.15); color: var(--signal);
    font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;
}

/* ---------- Expander cards ---------- */
div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    background: var(--panel) !important;
    overflow: hidden;
}

/* ---------- Result verdict cards ---------- */
.result-safe {
    background: rgba(52, 211, 153, 0.08); border: 1px solid rgba(52, 211, 153, 0.35);
    border-left: 4px solid var(--safe);
    padding: 1.3rem 1.5rem; border-radius: 12px; margin-top: 1rem;
}
.result-safe h3 { color: var(--safe) !important; margin: 0 0 0.3rem 0; font-size: 1.25rem; }
.result-safe p { color: var(--text) !important; margin: 0; }

.result-danger {
    background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.35);
    border-left: 4px solid var(--danger);
    padding: 1.3rem 1.5rem; border-radius: 12px; margin-top: 1rem;
}
.result-danger h3 { color: var(--danger) !important; margin: 0 0 0.3rem 0; font-size: 1.25rem; }
.result-danger p { color: var(--text) !important; margin: 0; }

/* ---------- Radial threat-level dial (signature element) ---------- */
.threat-dial-wrap { display: flex; align-items: center; gap: 1.6rem; margin-top: 0.9rem; }
.threat-dial {
    width: 110px; height: 110px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.threat-dial-inner {
    width: 84px; height: 84px; border-radius: 50%;
    background: var(--ink);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.threat-dial-pct {
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.3rem;
    color: var(--text); line-height: 1;
}
.threat-dial-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
    color: var(--muted); letter-spacing: 0.05em; margin-top: 0.2rem;
}
.threat-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--muted); }
.threat-meta b { color: var(--text); }

/* ---------- Red-flag chips ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.8rem; border-radius: 999px; font-size: 0.82rem;
    font-family: 'Inter', sans-serif;
}
.chip-danger { background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.4); color: #FCA5B1; }
.chip-safe { background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52, 211, 153, 0.4); color: #6EE7B7; }

/* ---------- Buttons ---------- */
.stButton>button {
    width: 100%; border-radius: 10px; height: 3.1em;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem;
    letter-spacing: 0.01em;
    background: linear-gradient(135deg, #2DD4BF 0%, #14B8A6 100%);
    color: #05201C; border: none;
    transition: filter 0.15s ease;
}
.stButton>button:hover { filter: brightness(1.08); color: #05201C; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--line);
}
.sidebar-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.95rem;
    color: var(--text); margin: 0.4rem 0 0.6rem 0;
    text-transform: uppercase; letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️</div>
    <h1>Phishing Website Detector</h1>
    <p>URL forensics scanner — enter website signals below and get an instant Phishing / Legitimate verdict, backed by a trained ML classifier.</p>
    <div class="scan-line"></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
model_choice = st.sidebar.radio(
    "Choose model",
    ["Random Forest", "Logistic Regression"],
    help="Both were trained on the same dataset; compare their predictions.",
    label_visibility="collapsed",
)
model = rf_model if model_choice == "Random Forest" else lr_model

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
st.sidebar.info(
    "This tool predicts whether a website is likely **Phishing** or "
    "**Legitimate** using real URL, domain-trust, and DNS-based features "
    "(sourced from a published academic dataset), trained with Logistic "
    "Regression / Random Forest classifiers."
)

# ---------------------------------------------------------------------
# QUICK HELPER: auto-fill a couple of fields from a pasted URL string
# ---------------------------------------------------------------------
st.markdown('<div class="section-label"><span class="badge">1</span> Quick URL scan (optional)</div>', unsafe_allow_html=True)
url_input = st.text_input(
    "Paste a URL here to auto-fill some fields below",
    placeholder="e.g. http://192.168.1.1/secure-login@paypal.com",
    label_visibility="collapsed",
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

st.markdown('<div class="section-label"><span class="badge">2</span> Website / URL characteristics</div>', unsafe_allow_html=True)

with st.expander("🔗  URL & domain structure", expanded=True):
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

with st.expander("🔒  Certificate & DNS trust signals", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        https_valid = st.selectbox("Valid HTTPS with trusted certificate?",
                                    ["No", "Yes"], index=auto_has_https)
        has_spf = st.selectbox("Domain has a valid SPF (anti-spoofing email) record?",
                                ["No", "Yes"], index=1)
    with col4:
        has_dns = st.selectbox("Domain has valid DNS nameservers?", ["No", "Yes"], index=1)
        num_mx = st.slider("Number of mail servers (MX records)", 0, 10, 2)

with st.expander("🌐  Domain age & network behaviour", expanded=True):
    col5, col6 = st.columns(2)
    with col5:
        domain_age_days = st.slider("Domain age (days since registration)", 0, 10000, 365)
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

if st.button("🔍  Run Scan"):
    input_df = pd.DataFrame([input_dict], columns=FEATURES)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # P(phishing)

    gauge_pct = round(proba * 100, 1)
    dial_color = "#F43F5E" if proba >= 0.5 else "#34D399"
    sweep_deg = gauge_pct * 3.6  # 0-100% -> 0-360deg

    # Radial dial built with a conic-gradient — no leading indentation on
    # these lines (Markdown treats 4+ leading spaces as a code block,
    # which causes stray closing tags to render as literal visible text).
    dial_html = (
        f'<div class="threat-dial-wrap">'
        f'<div class="threat-dial" style="background: conic-gradient({dial_color} {sweep_deg}deg, #1a2332 {sweep_deg}deg);">'
        f'<div class="threat-dial-inner">'
        f'<div class="threat-dial-pct">{gauge_pct:.0f}%</div>'
        f'<div class="threat-dial-label">RISK</div>'
        f'</div></div>'
        f'<div class="threat-meta">Model: <b>{model_choice}</b><br>Phishing probability: <b>{proba:.1%}</b><br>Legitimate probability: <b>{1 - proba:.1%}</b></div>'
        f'</div>'
    )

    if prediction == 1:
        st.markdown(
            f'<div class="result-danger">'
            f'<h3>⚠️  Likely PHISHING website</h3>'
            f'<p>This URL shows structural patterns consistent with phishing attempts.</p>'
            f'{dial_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.warning("Do not enter personal information or credentials on this site.")
    else:
        st.markdown(
            f'<div class="result-safe">'
            f'<h3>✅  Likely LEGITIMATE website</h3>'
            f'<p>No strong phishing indicators detected in this input.</p>'
            f'{dial_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.success("This site's signals align with typical legitimate websites.")

    with st.expander("See the exact feature values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # -----------------------------------------------------------------
    # RED-FLAG CHECKLIST — simple rule-based explanation, independent of
    # the model, so a viewer instantly sees *which* raw signals looked risky
    # -----------------------------------------------------------------
    st.markdown('<div class="section-label"><span class="badge">🚩</span> Signals detected in this input</div>', unsafe_allow_html=True)
    flags = [
        (input_dict["Having_At_Symbol"] == 1, "Contains '@' symbol"),
        (input_dict["Having_Hyphen_In_Domain"] == 1, "Hyphen in domain"),
        (input_dict["Num_Dots_In_Domain"] >= 3, "3+ dots in domain"),
        (input_dict["Is_Shortened_URL"] == 1, "URL shortener used"),
        (input_dict["Has_Email_In_URL"] == 1, "Email address in URL"),
        (input_dict["Has_Valid_SSL"] == 0, "No valid HTTPS"),
        (input_dict["Has_SPF_Record"] == 0, "No SPF record"),
        (input_dict["Has_DNS_Nameservers"] == 0, "No DNS nameservers"),
        (input_dict["Domain_Age_Days"] < 180, "Domain < 6 months old"),
        (input_dict["Num_Redirects"] >= 3, "3+ redirects"),
    ]
    triggered = [msg for cond, msg in flags if cond]
    clean = [msg for cond, msg in flags if not cond]

    chip_html = '<div class="chip-row">'
    for msg in triggered:
        chip_html += f'<span class="chip chip-danger">⚠ {msg}</span>'
    for msg in clean:
        chip_html += f'<span class="chip chip-safe">✓ {msg}</span>'
    chip_html += '</div>'
    st.markdown(chip_html, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # FEATURE IMPORTANCE — shows which features the MODEL globally relies
    # on most, for interpretability during the viva
    # -----------------------------------------------------------------
    st.markdown('<div class="section-label"><span class="badge">📊</span> What the model weighs most heavily</div>', unsafe_allow_html=True)
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
