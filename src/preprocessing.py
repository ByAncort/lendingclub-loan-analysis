import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
INTERIM_DIR = "data/interim"
EXTERNAL_DIR = "data/external"
SAMPLE_SIZE = 20000

TARGET_MAP = {
    "Fully Paid": 0,
    "Current": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
    "In Grace Period": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

DROP_COLS = [
    "id", "member_id", "url", "desc", "title",
    "zip_code", "emp_title",
    "hardship_type", "hardship_reason", "hardship_status",
    "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date",
    "debt_settlement_flag_date", "settlement_status",
    "settlement_date",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "next_pymnt_d",
    "hardship_loan_status",
]

def _sample_name(n):
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)

def load_accepted(sample=True):
    fname = f"accepted_sample_{_sample_name(SAMPLE_SIZE)}.csv" if sample else "accepted_2007_to_2018Q4.csv"
    base = INTERIM_DIR if sample else RAW_DIR
    path = os.path.join(base, fname)
    logger.info(f"Loading accepted: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Shape: {df.shape}")
    return df

def load_rejected(sample=True):
    path = os.path.join(INTERIM_DIR, "rejected_sample_5k.csv") if sample else os.path.join(RAW_DIR, "rejected_2007_to_2018Q4.csv")
    logger.info(f"Loading rejected: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Shape: {df.shape}")
    return df

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def parse_money(col):
    if col.dtype == object:
        return col.replace(r'[\$,]', '', regex=True).astype(float)
    return col

def parse_percent(col):
    if col.dtype == object:
        return col.str.rstrip("%").astype(float) / 100
    return col

def parse_term(col):
    if col.dtype == object:
        return col.str.extract(r"(\d+)", expand=False).astype(float)
    return col

def create_target(df):
    df["bad_loan"] = df["loan_status"].map(TARGET_MAP)
    df = df.dropna(subset=["bad_loan"])
    df["bad_loan"] = df["bad_loan"].astype(int)
    return df

def parse_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    return df

def flag_hardship(df):
    if "hardship_flag" in df.columns:
        df["had_hardship"] = (df["hardship_flag"] == "Y").astype(int)
    else:
        df["had_hardship"] = 0
    return df

def flag_debt_settlement(df):
    if "debt_settlement_flag" in df.columns:
        df["debt_settlement"] = (df["debt_settlement_flag"] == "Y").astype(int)
    else:
        df["debt_settlement"] = 0
    return df

def calc_fico_mid(df):
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df

def preprocess_accepted(df):
    logger.info("Preprocessing accepted loans...")
    df = clean_column_names(df)
    df = create_target(df)
    df = parse_dates(df, ["issue_d", "earliest_cr_line", "last_pymnt_d",
                          "last_credit_pull_d", "sec_app_earliest_cr_line"])
    df = calc_fico_mid(df)
    df = flag_hardship(df)
    df = flag_debt_settlement(df)
    df["term"] = parse_term(df["term"])
    df["int_rate"] = parse_percent(df["int_rate"])
    df["revol_util"] = parse_percent(df["revol_util"])
    money_cols = ["loan_amnt", "funded_amnt", "funded_amnt_inv",
                  "annual_inc", "installment", "total_pymnt",
                  "total_rec_prncp", "total_rec_int", "recoveries",
                  "collection_recovery_fee", "last_pymnt_amnt",
                  "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
                  "avg_cur_bal", "bc_open_to_buy", "tot_hi_cred_lim",
                  "total_bal_ex_mort", "total_bc_limit",
                  "total_il_high_credit_limit", "annual_inc_joint",
                  "revol_bal", "out_prncp", "out_prncp_inv",
                  "total_pymnt_inv", "total_rec_late_fee",
                  "max_bal_bc", "total_bal_il", "hardship_amount",
                  "settlement_amount", "revol_bal_joint"]
    for col in money_cols:
        if col in df.columns:
            df[col] = parse_money(df[col])
    df["emp_length"] = df["emp_length"].str.extract(r"(\d+)", expand=False).astype(float)
    df["mths_since_last_delinq"] = df["mths_since_last_delinq"].replace(0, np.nan)
    df["mths_since_last_record"] = df["mths_since_last_record"].replace(0, np.nan)
    df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0).astype(int)
    logger.info(f"Shape after preprocessing: {df.shape}")
    logger.info(f"Target distribution:\n{df['bad_loan'].value_counts()}")
    return df

def preprocess_rejected(df):
    logger.info("Preprocessing rejected loans...")
    df = clean_column_names(df)
    date_cols = [c for c in df.columns if "date" in c]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
    if "amount_requested" in df.columns:
        df["amount_requested"] = parse_money(df["amount_requested"])
    target_dti_col = [c for c in df.columns if "debt" in c and "income" in c]
    if target_dti_col:
        df["dti"] = df[target_dti_col[0]].str.rstrip("%").astype(float)
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["rejected"] = 1
    logger.info(f"Shape after preprocessing: {df.shape}")
    return df

def remove_leakage(df):
    drop = [col for col in DROP_COLS if col.replace(" ", "_").lower() in df.columns]
    existing_drop = [c for c in drop if c in df.columns]
    post_loan_cols = [c for c in df.columns if c.startswith(("out_prncp", "total_pymnt", "total_rec",
                                                              "last_pymnt", "last_fico", "recoveries",
                                                              "collection_", "next_", "hardship_",
                                                              "settlement_", "debt_settlement_"))]
    all_drop = list(set(existing_drop + post_loan_cols))
    df = df.drop(columns=all_drop, errors="ignore")
    logger.info(f"Dropped {len(all_drop)} leakage columns")
    return df

def load_fred():
    path = os.path.join(EXTERNAL_DIR, "fred_indicators.csv")
    if not os.path.exists(path):
        logger.warning("FRED indicators not found. Run scripts/fetch_fred.py first.")
        return None
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    logger.info(f"Loaded FRED indicators: {df.shape}")
    return df

def merge_fred(df, fred_df):
    if fred_df is None:
        return df
    df["year"] = df["issue_d"].dt.year
    df["month"] = df["issue_d"].dt.month
    before = df.shape[1]
    df = df.merge(fred_df[["year", "month", "unrate", "fed_funds", "cpi"]],
                  on=["year", "month"], how="left")
    logger.info(f"FRED columns added. Shape: {df.shape} (was {before} cols)")
    return df

def save_processed(df, name):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, name)
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path} ({df.shape})")
    return path

def run_pipeline(sample=True):
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    df_acc = load_accepted(sample)
    df_acc = preprocess_accepted(df_acc)
    df_acc = remove_leakage(df_acc)
    fred_df = load_fred()
    df_acc = merge_fred(df_acc, fred_df)

    df_rej = load_rejected(sample)
    df_rej = preprocess_rejected(df_rej)

    save_processed(df_acc, "accepted_clean.csv")
    save_processed(df_rej, "rejected_clean.csv")

    rej_cols = {}
    if "amount_requested" in df_rej.columns: rej_cols["loan_amnt"] = df_rej["amount_requested"]
    if "dti" in df_rej.columns: rej_cols["dti"] = df_rej["dti"]
    if "risk_score" in df_rej.columns: rej_cols["fico_score"] = df_rej["risk_score"]
    if "state" in df_rej.columns: rej_cols["addr_state"] = df_rej["state"]
    if "employment_length" in df_rej.columns: rej_cols["emp_length"] = df_rej["employment_length"]
    if "application_date" in df_rej.columns: rej_cols["issue_d"] = df_rej["application_date"]
    rej_cols["rejected"] = 1
    df_rej_summary = pd.DataFrame(rej_cols)

    df_acc_summary = df_acc[["loan_amnt", "dti", "fico_score", "addr_state",
                             "emp_length", "issue_d", "bad_loan"]].copy()
    df_acc_summary["rejected"] = 0
    if "int_rate" in df_acc.columns:
        df_acc_summary["int_rate"] = df_acc["int_rate"]
    if "grade" in df_acc.columns:
        df_acc_summary["grade"] = df_acc["grade"]
    if "annual_inc" in df_acc.columns:
        df_acc_summary["annual_inc"] = df_acc["annual_inc"]
    if "home_ownership" in df_acc.columns:
        df_acc_summary["home_ownership"] = df_acc["home_ownership"]

    combined = pd.concat([df_acc_summary, df_rej_summary], ignore_index=True, sort=False)
    save_processed(combined, "combined_summary.csv")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_pipeline(sample=True)
