"""
generate_dataset.py
--------------------
Builds phishing_dataset.csv from a REAL, published academic dataset.

DATA SOURCE (real, not synthetic):
    G. Vrbancic, I. Fister Jr., V. Podgorelec.
    "Datasets for Phishing Websites Detection." Data in Brief, Vol. 33, 2020.
    Repository: https://github.com/GregaVrbancic/Phishing-Dataset
    File used: dataset_small.csv (a curated, class-balanced subset of the
    authors' full 88,000+ instance dataset -- 111 raw technical features
    extracted from real crawled URLs, each labelled phishing (1) or
    legitimate (0) by the original researchers).

WHAT THIS SCRIPT DOES:
    raw_dataset_small.csv (bundled in this project folder) contains a
    114-row real sample of that published dataset with its original 111
    cryptic engineering column names (e.g. qty_at_url, tls_ssl_certificate,
    time_domain_activation). This script:
      1. Loads that raw real data
      2. Selects 16 of the most interpretable, viva-friendly features
      3. Renames/derives them into clear, human-readable columns
      4. Saves the result as phishing_dataset.csv, which the rest of the
         pipeline (phishing_detection.py) reads and trains on.

NOTE ON SAMPLE SIZE: this bundled sample has 114 real rows (a subset,
since this project was built in a sandbox without live internet access to
pull the full file). For a larger, more robust model, download the full
dataset_small.csv (~58,000 rows) or dataset_full.csv (~88,000 rows)
directly from the GitHub link above and re-run this script by simply
overwriting raw_dataset_small.csv with the full file (same column names).
"""

import numpy as np
import pandas as pd

RAW_FILE = "raw_dataset_small.csv"


def build_clean_dataset(raw_path=RAW_FILE):
    raw = pd.read_csv(raw_path)

    df = pd.DataFrame()

    # --- URL-structure features ---
    df["URL_Length"] = raw["length_url"]
    df["Domain_Length"] = raw["domain_length"]
    df["Having_At_Symbol"] = (raw["qty_at_url"] > 0).astype(int)
    df["Having_Hyphen_In_Domain"] = (raw["qty_hyphen_domain"] > 0).astype(int)
    df["Num_Dots_In_Domain"] = raw["qty_dot_domain"]  # proxy for subdomain count
    df["Is_Shortened_URL"] = (raw["url_shortened"] > 0).astype(int)
    df["Num_Slashes_In_URL"] = raw["qty_slash_url"]
    df["Has_Email_In_URL"] = (raw["email_in_url"] > 0).astype(int)

    # --- Certificate / domain-trust features ---
    df["Has_Valid_SSL"] = (raw["tls_ssl_certificate"] > 0).astype(int)
    df["Has_SPF_Record"] = (raw["domain_spf"] > 0).astype(int)
    df["Has_DNS_Nameservers"] = (raw["qty_nameservers"] > 0).astype(int)
    df["Num_Mail_Servers"] = raw["qty_mx_servers"].clip(lower=0)

    # Domain age / expiry -- raw values are in days; -1 in the source data
    # means "could not be resolved" -> treat as 0 (unknown/very new)
    df["Domain_Age_Days"] = raw["time_domain_activation"].clip(lower=0).fillna(0)
    df["Domain_Expiry_Days"] = raw["time_domain_expiration"].clip(lower=0).fillna(0)
    df["TTL_Hostname"] = raw["ttl_hostname"].clip(lower=0).fillna(0)

    # --- Behavioural / traffic features ---
    df["Num_Redirects"] = raw["qty_redirects"].clip(lower=0)
    df["Server_Response_Time"] = raw["time_response"].clip(lower=0).fillna(0)
    df["Num_IPs_Resolved"] = raw["qty_ip_resolved"].clip(lower=0)

    # --- Target ---
    df["Result"] = raw["phishing"].astype(int)  # 1 = Phishing, 0 = Legitimate

    return df


if __name__ == "__main__":
    dataset = build_clean_dataset()
    dataset.to_csv("phishing_dataset.csv", index=False)
    print(f"Real dataset saved: phishing_dataset.csv | shape = {dataset.shape}")
    print(f"Class balance:\n{dataset['Result'].value_counts()}")
