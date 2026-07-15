"""
Pipeline de limpieza para el dataset de Lending Club
(https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Refactor sobre la version original: agrega copias explicitas, logging
de perdida de filas, validacion de columnas, y separa mejor que
funcion opera sobre que dataframe.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("lc_pipeline")


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def _require_columns(df, cols, fn_name):
    """Valida que las columnas necesarias existan antes de operar."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning(f"[{fn_name}] Columnas faltantes, se omiten: {missing}")
    return [c for c in cols if c in df.columns]


def _log_shape_change(fn_name, before, after):
    lost = before - after
    if lost > 0:
        pct = lost / before * 100 if before else 0
        log.info(f"[{fn_name}] Filas: {before} -> {after} ({lost} eliminadas, {pct:.2f}%)")


# ---------------------------------------------------------------------
# FRED (datos macroeconomicos)
# ---------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL_DIR = os.path.join(_BASE_DIR, "data", "external")

def _load_fred():
    path = os.path.join(_EXTERNAL_DIR, "fred_indicators.csv")
    if not os.path.exists(path):
        log.warning(f"[FRED] Archivo no encontrado: {path}")
        return None
    fred = pd.read_csv(path)
    fred["date"] = pd.to_datetime(fred["date"])
    fred["_ym"] = fred["date"].dt.to_period("M").astype(str)
    log.info(f"[FRED] Cargados {len(fred)} registros")
    return fred

def merge_fred_columns(df_acc, fred_df):
    if fred_df is None:
        return df_acc
    df_acc = df_acc.copy()
    df_acc["_ym"] = df_acc["issue_d"].dt.to_period("M").astype(str)
    before = df_acc.shape[1]
    df_acc = df_acc.merge(fred_df[["_ym", "unrate", "fed_funds", "cpi"]], on="_ym", how="left")
    df_acc.drop(columns=["_ym"], inplace=True)
    log.info(f"[merge_fred_columns] {df_acc.shape[1] - before} columnas FRED agregadas")
    return df_acc


# ---------------------------------------------------------------------
# Normalizacion de nombres / strings
# ---------------------------------------------------------------------

def normalize_columns(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_"))
    return df


def clean_string_whitespace(df):
    df = df.copy()
    str_cols = df.select_dtypes("object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()
    return df


# ---------------------------------------------------------------------
# Target (solo aplica a aceptados)
# ---------------------------------------------------------------------

def build_target(df_acc):
    df_acc = df_acc.copy()
    n0 = len(df_acc)

    if "loan_status" not in df_acc.columns:
        raise KeyError("build_target requiere la columna 'loan_status'")

    TARGET_MAP = {
        "Fully Paid": 0, "Current": 0,
        "Charged Off": 1, "Default": 1,
        "Late (31-120 days)": 1, "Late (16-30 days)": 1,
        "In Grace Period": 1,
        "Does not meet the credit policy. Status:Fully Paid": 0,
        "Does not meet the credit policy. Status:Charged Off": 1,
    }
    df_acc["bad_loan"] = df_acc["loan_status"].map(TARGET_MAP)
    df_acc = df_acc.dropna(subset=["bad_loan"])
    df_acc["bad_loan"] = df_acc["bad_loan"].astype(int)

    n1 = len(df_acc)
    _log_shape_change("build_target", n0, n1)
    log.info(f"[build_target] bad_loan = {df_acc['bad_loan'].sum()} malos "
              f"({df_acc['bad_loan'].mean()*100:.1f}%)")
    return df_acc


# ---------------------------------------------------------------------
# Parseo de tipos (aceptados)
# ---------------------------------------------------------------------

def parse_dates(df_acc):
    df_acc = df_acc.copy()
    date_cols = ["issue_d", "earliest_cr_line", "last_pymnt_d",
                 "last_credit_pull_d", "sec_app_earliest_cr_line"]
    for col in _require_columns(df_acc, date_cols, "parse_dates"):
        df_acc[col] = pd.to_datetime(df_acc[col], format="%b-%Y", errors="coerce")
    return df_acc


def parse_percentages(df_acc):
    df_acc = df_acc.copy()
    pct_cols = ["int_rate", "revol_util"]
    for col in _require_columns(df_acc, pct_cols, "parse_percentages"):
        if not pd.api.types.is_numeric_dtype(df_acc[col]):
            df_acc[col] = df_acc[col].str.replace("%", "", regex=False).astype(float)
    return df_acc


def parse_money(df_acc):
    df_acc = df_acc.copy()
    money_cols = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "annual_inc",
        "installment", "revol_bal", "total_rev_hi_lim", "tot_cur_bal",
        "avg_cur_bal", "bc_open_to_buy", "tot_hi_cred_lim",
        "total_bal_ex_mort", "total_bc_limit", "max_bal_bc",
        "total_il_high_credit_limit", "annual_inc_joint",
        "revol_bal_joint", "tot_coll_amt", "total_bal_il",
    ]
    for col in _require_columns(df_acc, money_cols, "parse_money"):
        if df_acc[col].dtype == object:
            df_acc[col] = df_acc[col].replace(r'[\$,]', '', regex=True).astype(float)
    return df_acc


def parse_term_and_employment(df_acc):
    """
    OJO: la version original usaba str.rstrip(" months") / rstrip("+ years"),
    que NO elimina el sufijo como string sino caracteres sueltos.
    Aca se usa str.replace, que es lo correcto.
    """
    df_acc = df_acc.copy()

    if "term" in df_acc.columns and df_acc["term"].dtype == object:
        df_acc["term"] = (df_acc["term"]
                           .str.replace(" months", "", regex=False)
                           .astype(int))

    if "emp_length" in df_acc.columns and df_acc["emp_length"].dtype == object:
        df_acc["emp_length"] = (df_acc["emp_length"]
                                 .str.replace("10+ years", "10", regex=False)
                                 .str.replace("< 1 year", "0", regex=False)
                                 .str.replace(" years", "", regex=False)
                                 .str.replace(" year", "", regex=False)
                                 .astype(float))
    return df_acc


# ---------------------------------------------------------------------
# Feature engineering (aceptados)
# ---------------------------------------------------------------------

def feature_engineering(df_acc):
    df_acc = df_acc.copy()

    if "fico_range_low" in df_acc.columns and "fico_range_high" in df_acc.columns:
        df_acc["fico_score"] = (df_acc["fico_range_low"] + df_acc["fico_range_high"]) / 2

    df_acc["had_hardship"] = (
        (df_acc["hardship_flag"] == "Y").astype(int)
        if "hardship_flag" in df_acc.columns else 0
    )
    df_acc["debt_settlement"] = (
        (df_acc["debt_settlement_flag"] == "Y").astype(int)
        if "debt_settlement_flag" in df_acc.columns else 0
    )
    return df_acc


def handle_missing_values(df_acc):
    df_acc = df_acc.copy()

    if "mths_since_last_delinq" in df_acc.columns:
        df_acc["had_delinquency"] = (df_acc["mths_since_last_delinq"].notna() & (df_acc["mths_since_last_delinq"] > 0)).astype(int)
        df_acc["mths_since_last_delinq"] = df_acc["mths_since_last_delinq"].replace(0, np.nan)

    for col in _require_columns(
        df_acc, ["mths_since_last_record"],
        "handle_missing_values"
    ):
        df_acc[col] = df_acc[col].replace(0, np.nan)

    if "pub_rec_bankruptcies" in df_acc.columns:
        df_acc["pub_rec_bankruptcies"] = df_acc["pub_rec_bankruptcies"].fillna(0).astype(int)

    return df_acc


# ---------------------------------------------------------------------
# Leakage (aceptados)
# ---------------------------------------------------------------------

def drop_leakage_columns(df_acc):
    df_acc = df_acc.copy()

    DROP_COLS = [
        "id", "member_id", "url", "desc", "title", "zip_code", "emp_title",
        "hardship_type", "hardship_reason", "hardship_status",
        "hardship_start_date", "hardship_end_date",
        "payment_plan_start_date", "debt_settlement_flag_date", "settlement_status",
        "settlement_date", "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount", "hardship_last_payment_amount",
        "next_pymnt_d", "hardship_loan_status",
    ]
    post_loan_prefixes = [
        "out_prncp", "total_pymnt", "total_rec",
        "last_pymnt", "last_fico", "recoveries",
        "collection_", "next_", "hardship_",
        "settlement_", "debt_settlement_",
    ]
    explicit_drop = [c for c in DROP_COLS if c in df_acc.columns]
    prefix_drop = [c for c in df_acc.columns if any(c.startswith(p) for p in post_loan_prefixes)]
    to_drop = sorted(set(explicit_drop + prefix_drop))

    df_acc = df_acc.drop(columns=to_drop, errors="ignore")
    log.info(f"[drop_leakage_columns] {len(to_drop)} columnas eliminadas")
    return df_acc


# ---------------------------------------------------------------------
# Rechazados
# ---------------------------------------------------------------------

def preprocess_rejected(df_rej):
    df_rej = df_rej.copy()

    date_cols = [c for c in df_rej.columns if "date" in c]
    for col in date_cols:
        df_rej[col] = pd.to_datetime(df_rej[col], format="%m/%d/%Y", errors="coerce")

    if "amount_requested" in df_rej.columns and df_rej["amount_requested"].dtype == object:
        df_rej["amount_requested"] = (
            df_rej["amount_requested"].replace(r'[\$,]', '', regex=True).astype(float)
        )

    dti_candidates = [c for c in df_rej.columns if "debt" in c and "income" in c]
    if dti_candidates:
        source_col = dti_candidates[0]
        df_rej["dti"] = df_rej[source_col].str.replace("%", "", regex=False).astype(float)
        df_rej["dti"] = df_rej["dti"].clip(upper=200.0)
        df_rej = df_rej.drop(columns=[source_col])

    if "risk_score" in df_rej.columns:
        df_rej["risk_score"] = pd.to_numeric(df_rej["risk_score"], errors="coerce")

    df_rej["rejected"] = 1
    return df_rej


# ---------------------------------------------------------------------
# Combinacion (para modelar aceptado vs rechazado)
# ---------------------------------------------------------------------

def build_combined_summary(df_acc, df_rej):
    """
    Devuelve un dataframe RESUMEN (no reemplaza a df_acc completo) pensado
    para modelar la decision de aceptar/rechazar. Para modelar bad_loan,
    usa df_acc completo por separado.
    """
    rej_map = {
        "amount_requested": "loan_amnt",
        "dti": "dti",
        "risk_score": "fico_score",
        "state": "addr_state",
        "employment_length": "emp_length",
        "application_date": "issue_d",
    }
    rej_cols = {new: df_rej[old] for old, new in rej_map.items() if old in df_rej.columns}
    rej_cols["rejected"] = 1
    df_rej_summary = pd.DataFrame(rej_cols)

    base_acc_cols = ["loan_amnt", "dti", "fico_score", "addr_state",
                      "emp_length", "issue_d", "bad_loan"]
    extra_cols = [c for c in ["int_rate", "grade", "annual_inc", "home_ownership"]
                  if c in df_acc.columns]
    acc_cols = _require_columns(df_acc, base_acc_cols, "build_combined_summary") + extra_cols

    df_acc_summary = df_acc[acc_cols].copy()
    df_acc_summary["rejected"] = 0

    df_comb = pd.concat([df_acc_summary, df_rej_summary], ignore_index=True, sort=False)
    log.info(f"[build_combined_summary] combinado: {len(df_comb)} filas "
              f"({len(df_acc_summary)} aceptados + {len(df_rej_summary)} rechazados)")
    return df_comb


# ---------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------

def clean_pipeline(df_acc, df_rej):
    """
    Corre todos los pasos de limpieza en orden sobre copias de los
    dataframes originales. Devuelve (df_acc_limpio, df_rej_limpio).
    """
    df_acc = normalize_columns(df_acc)
    df_rej = normalize_columns(df_rej)

    df_acc = clean_string_whitespace(df_acc)
    df_rej = clean_string_whitespace(df_rej)

    df_acc = build_target(df_acc)
    df_acc = parse_dates(df_acc)
    df_acc = parse_percentages(df_acc)
    df_acc = parse_money(df_acc)
    df_acc = parse_term_and_employment(df_acc)
    df_acc = feature_engineering(df_acc)
    df_acc = handle_missing_values(df_acc)
    df_acc = drop_leakage_columns(df_acc)
    fred_df = _load_fred()
    df_acc = merge_fred_columns(df_acc, fred_df)

    df_rej = preprocess_rejected(df_rej)

    return df_acc, df_rej
