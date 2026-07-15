import streamlit as st
import pandas as pd
import numpy as np
from dashboard.utils.model_loader import load_model, load_metadata
from dashboard.utils.data_loader import load_accepted

FEATURE_DEFAULTS = {
    "term": 36.0, "emp_length": 6.0, "annual_inc": 65000.0, "dti": 18.0,
    "delinq_2yrs": 0.0, "fico_range_low": 690.0, "inq_last_6mths": 0.0,
    "open_acc": 11.0, "pub_rec": 0.0, "revol_util": 50.0, "total_acc": 22.0,
    "acc_open_past_24mths": 3.0, "avg_cur_bal": 13000.0,
    "bc_open_to_buy": 8000.0, "bc_util": 55.0, "chargeoff_within_12_mths": 0.0,
    "delinq_amnt": 0.0, "mo_sin_old_il_acct": 80.0,
    "mo_sin_old_rev_tl_op": 150.0, "mo_sin_rcnt_rev_tl_op": 12.0,
    "mo_sin_rcnt_tl": 18.0, "mort_acc": 1.0, "mths_since_last_delinq": 60.0,
    "mths_since_last_record": 60.0, "mths_since_recent_bc": 10.0,
    "mths_since_recent_bc_dlq": 40.0, "mths_since_recent_inq": 6.0,
    "mths_since_recent_revol_delinq": 40.0, "num_accts_ever_120_pd": 0.0,
    "num_actv_bc_tl": 3.0, "num_actv_rev_tl": 7.0,
    "num_bc_sats": 5.0, "num_bc_tl": 3.0, "num_il_tl": 5.0,
    "num_op_rev_tl": 6.0, "num_rev_accts": 9.0, "num_rev_tl_bal_gt_0": 5.0,
    "num_sats": 15.0, "num_tl_120dpd_2m": 0.0, "num_tl_30dpd": 0.0,
    "num_tl_90g_dpd_24m": 0.0, "num_tl_op_past_12m": 3.0,
    "pct_tl_nvr_dlq": 90.0, "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0, "tot_cur_bal": 30000.0, "tot_hi_cred_lim": 60000.0,
    "total_bal_ex_mort": 20000.0, "total_bc_limit": 15000.0,
    "total_il_high_credit_limit": 20000.0, "revol_bal": 10000.0,
    "installment": 380.0, "funded_amnt": 13000.0, "funded_amnt_inv": 13000.0,
    "fico_range_high": 694.0, "last_fico_range_high": 694.0,
    "last_fico_range_low": 690.0, "collections_12_mths_ex_med": 0.0,
    "debt_settlement_flag": 0.0, "hardship_flag": 0.0,
    "hardship_length": 0.0, "hardship_dpd": 0.0,
    "hardship_loan_status": 0.0, "hardship_payoff_balance_amount": 0.0,
    "hardship_amount": 0.0, "disbursement_method": 0.0,
    "deferral_term": 0.0, "settlement_status": 0.0,
    "settlement_amount": 0.0, "settlement_percentage": 0.0,
    "settlement_term": 0.0,
}

CATEGORICAL_DEFAULTS = {
    "grade": "C", "sub_grade": "C1", "home_ownership": "RENT",
    "verification_status": "Not Verified", "purpose": "debt_consolidation",
    "addr_state": "CA", "initial_list_status": "f", "application_type": "Individual",
    "pymnt_plan": "n", "policy_code": 1.0,
}

GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
SUB_GRADES = [f"{g}{n}" for g in GRADE_ORDER for n in range(1, 6)]
OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "ANY", "NONE"]
PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "car", "medical", "small_business", "other", "house", "moving",
    "vacation", "wedding", "renewable_energy", "educational",
]
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
          "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
          "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT",
          "IA", "NV", "AR", "KS", "MS", "NM", "NE", "WV", "ID", "HI",
          "NH", "ME", "MT", "RI", "DE", "SD", "ND", "AK", "VT", "WY"]

_RANGES = {
    "loan_amnt": (1000, 40000), "int_rate": (5.0, 31.0),
    "annual_inc": (10000, 300000), "dti": (0, 50),
    "fico_range_low": (600, 850), "revol_util": (0, 120),
    "open_acc": (1, 40), "total_acc": (2, 80),
    "delinq_2yrs": (0, 10), "inq_last_6mths": (0, 15),
    "emp_length": (0, 10), "pub_rec": (0, 5),
    "mort_acc": (0, 10), "avg_cur_bal": (0, 200000),
    "bc_open_to_buy": (0, 50000), "bc_util": (0, 100),
    "tot_cur_bal": (0, 300000), "total_bc_limit": (0, 80000),
    "total_il_high_credit_limit": (0, 200000),
    "mo_sin_old_rev_tl_op": (0, 800), "revol_bal": (0, 100000),
    "num_sats": (0, 40), "pct_tl_nvr_dlq": (0, 100),
    "num_il_tl": (0, 20), "num_rev_accts": (0, 30),
    "num_actv_bc_tl": (0, 10), "num_actv_rev_tl": (0, 20),
    "num_bc_tl": (0, 10), "num_bc_sats": (0, 15),
    "num_op_rev_tl": (0, 15), "num_rev_tl_bal_gt_0": (0, 15),
    "num_tl_op_past_12m": (0, 15), "acc_open_past_24mths": (0, 15),
    "num_tl_30dpd": (0, 5), "num_tl_90g_dpd_24m": (0, 5),
    "num_accts_ever_120_pd": (0, 5),
    "mths_since_last_delinq": (0, 200), "mths_since_last_record": (0, 200),
    "mths_since_recent_bc": (0, 200), "mths_since_recent_inq": (0, 50),
    "mths_since_recent_revol_delinq": (0, 200),
    "chargeoff_within_12_mths": (0, 5),
    "collections_12_mths_ex_med": (0, 5), "pub_rec_bankruptcies": (0, 5),
    "tax_liens": (0, 5), "delinq_amnt": (0, 5000),
    "mo_sin_old_il_acct": (0, 400), "mo_sin_rcnt_rev_tl_op": (0, 100),
    "mo_sin_rcnt_tl": (0, 100), "mths_since_recent_bc_dlq": (0, 200),
    "tot_hi_cred_lim": (0, 300000), "total_bal_ex_mort": (0, 200000),
    "total_bc_limit": (0, 80000), "installment": (20, 2000),
    "funded_amnt": (1000, 40000), "funded_amnt_inv": (1000, 40000),
}

def _sample_to_df(form, sample_meta):
    row = {}
    for col, default in FEATURE_DEFAULTS.items():
        val = form.get(col, default)
        if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
            val = default
        row[col] = float(val)
    for col, default in CATEGORICAL_DEFAULTS.items():
        row[col] = form.get(col, default)
    row.update(sample_meta)
    return pd.DataFrame([row])

def show():
    st.title("Prediccion de Default")
    st.markdown("Ingrese los datos del solicitante para estimar su probabilidad de default.")

    model = load_model()
    meta = load_metadata()

    if model is None or meta is None:
        st.warning("Modelo no disponible.")
        return

    with st.expander("Metricas del modelo (test set)", expanded=False):
        m = meta["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
        c2.metric("Precisión", f"{m['precision']:.4f}")
        c3.metric("Recall", f"{m['recall']:.4f}")
        c4.metric("F1-Score", f"{m['f1']:.4f}")

    st.markdown("---")

    with st.form("pred_form"):
        st.subheader("Datos del Prestamo")
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            loan_amnt = st.number_input("Monto solicitado ($)", 1000, 40000, 13000, step=500)
            term = st.selectbox("Plazo (meses)", [36, 60], index=0)
            grade = st.selectbox("Grade", GRADE_ORDER, index=2)
            sub_grade = st.selectbox("Sub Grade", [f"{grade}{n}" for n in range(1, 6)], index=0)
            purpose = st.selectbox("Propósito", PURPOSES, index=0)

        with rc2:
            int_rate = st.slider("Tasa de interés (%)", 5.0, 31.0, 13.0, 0.1)
            annual_inc = st.number_input("Ingreso anual ($)", 0, 1000000, 65000, step=1000)
            emp_length = st.slider("Años de empleo", 0, 10, 6)
            home_ownership = st.selectbox("Tipo de vivienda", OWNERSHIP, index=0)
            dti = st.slider("DTI (%)", 0.0, 50.0, 18.0, 0.1)

        with rc3:
            fico = st.slider("FICO Score", 600, 850, 690)
            revol_util = st.slider("Revolving Utilización (%)", 0, 120, 50)
            delinq_2yrs = st.slider("Delincuencias (2 años)", 0, 10, 0)
            inq_last_6mths = st.slider("Consultas (6 meses)", 0, 15, 0)
            open_acc = st.slider("Cuentas abiertas", 1, 40, 11)

        st.subheader("Informacion Adicional")
        rc4, rc5 = st.columns(2)
        with rc4:
            verification_status = st.selectbox(
                "Verificación", ["Source Verified", "Not Verified", "Verified"], index=1
            )
            application_type = st.selectbox("Tipo aplicación", ["Individual", "Joint App"], index=0)
        with rc5:
            initial_list_status = st.selectbox("Listado inicial", ["f", "w"], index=0)
            addr_state = st.selectbox("Estado", STATES, index=0)

        submitted = st.form_submit_button("Predecir Riesgo", type="primary", use_container_width=True)

    if submitted:
        form_data = {
            "loan_amnt": loan_amnt, "term": float(term), "int_rate": int_rate,
            "grade": grade, "sub_grade": sub_grade, "purpose": purpose,
            "annual_inc": annual_inc, "emp_length": float(emp_length),
            "home_ownership": home_ownership, "dti": dti,
            "fico_range_low": float(fico), "fico_range_high": float(fico + 4),
            "last_fico_range_low": float(fico), "last_fico_range_high": float(fico + 4),
            "revol_util": revol_util, "delinq_2yrs": float(delinq_2yrs),
            "inq_last_6mths": float(inq_last_6mths), "open_acc": float(open_acc),
            "verification_status": verification_status,
            "application_type": application_type,
            "initial_list_status": initial_list_status,
            "addr_state": addr_state,
        }

        with st.spinner("Calculando probabilidad de default..."):
            sample = _sample_to_df(form_data, {"debt_settlement_flag": 0.0, "hardship_flag": 0.0})

            expected_cols = model.feature_names_in_
            for col in expected_cols:
                if col not in sample.columns:
                    sample[col] = 0.0
            sample = sample[expected_cols]

            proba = model.predict_proba(sample)[0, 1]
            pred_class = int(proba >= 0.5)

        st.markdown("---")
        st.subheader("Resultado")

        res1, res2, res3 = st.columns([1, 2, 1])
        with res2:
            delta = proba - meta["metrics"].get("default_rate_train", 0.137)
            st.metric(
                label="Probabilidad de Default",
                value=f"{proba:.1%}",
                delta=f"{delta:+.1%} vs población",
                delta_color="inverse",
            )

        if pred_class == 1:
            st.error("**ALTO RIESGO** — El modelo clasifica este prestamo como potencial default.")
        else:
            st.success("**BAJO RIESGO** — El modelo clasifica este prestamo como buen pagador.")

        st.progress(int(proba * 100), text=f"Riesgo relativo: {proba:.1%}")

        with st.expander("Interpretacion"):
            st.markdown(f"""
            - **Punto de corte:** 0.50
            - **Precisión (precision):** {m['precision']:.1%} de los préstamos marcados como malos realmente lo son
            - **Sensibilidad (recall):** {m['recall']:.1%} de los malos préstamos son detectados
            - **AUC-ROC:** {m['auc_roc']:.1%} (poder discriminativo general)
            """)
