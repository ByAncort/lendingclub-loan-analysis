import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
RANDOM_STATE = 42

df = pd.read_csv(os.path.join(PROCESSED_DIR, "accepted_clean.csv"), low_memory=False)

X = df.drop(columns=["bad_loan"])
y = df["bad_loan"]

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
remove_cols = [c for c in num_cols + cat_cols if c in ("loan_status", "issue_d",
    "earliest_cr_line", "last_credit_pull_d", "deferral_term",
    "disbursement_method", "pymnt_plan", "policy_code",
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog", "revol_bal_joint",
    "annual_inc_joint", "dti_joint", "verification_status_joint")]
num_cols = [c for c in num_cols if c not in remove_cols]
cat_cols = [c for c in cat_cols if c not in remove_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        max_depth=4, learning_rate=0.01, n_estimators=200,
        subsample=0.7, colsample_bytree=0.8, min_child_weight=3,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1
    )),
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
print("=== XGBoost Tuned ===")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
print(f"F1:       {f1_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Good", "Bad"]))

os.makedirs(MODELS_DIR, exist_ok=True)
path = os.path.join(MODELS_DIR, "xgboost_pipeline.pkl")
joblib.dump(pipeline, path)
print(f"Model saved to {path}")

metadata = {
    "features": {"num": num_cols, "cat": cat_cols},
    "metrics": {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    },
    "scale_pos_weight": scale_pos,
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "default_rate_train": float(y_train.mean()),
}
import json
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved")
