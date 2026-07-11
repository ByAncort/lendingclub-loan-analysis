import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 11})
OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

df_acc = pd.read_csv("data/processed/accepted_clean.csv")
df_rej = pd.read_csv("data/processed/rejected_clean.csv")

# 1. Target distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
tc = df_acc["bad_loan"].value_counts()
tp = df_acc["bad_loan"].value_counts(normalize=True) * 100
ax[0].bar(["Good (Paga)", "Bad (Default)"], tc.values, color=["#2ecc71", "#e74c3c"], width=0.5)
ax[0].set_ylabel("Count")
for i, v in enumerate(tc.values):
    ax[0].text(i, v + 20, f"{v:,}\n({tp.values[i]:.1f}%)", ha="center", fontweight="bold")
ax[1].pie(tp.values, labels=["Good", "Bad"], autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"], startangle=90, explode=(0, 0.05))
ax[1].set_title("Target Distribution")
plt.suptitle("Loan Default (Bad Loan) Distribution", fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/target_distribution.png", dpi=150, bbox_inches="tight")
plt.close(); print("1/10 - Target distribution")

# 2. Numerical distributions (10 features)
num_cols = ["loan_amnt", "annual_inc", "dti", "fico_score", "int_rate", "open_acc",
            "delinq_2yrs", "revol_util", "emp_length", "pub_rec"]
titles = ["Loan Amount", "Annual Income", "DTI", "FICO Score", "Interest Rate", "Open Accounts",
          "Delinq 2y", "Revol Util", "Emp Length", "Pub Rec"]
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, (col, title) in enumerate(zip(num_cols, titles)):
    d = df_acc[col].dropna()
    axes[i].hist(d, bins=50, edgecolor="white", alpha=0.75, color="steelblue")
    axes[i].axvline(d.median(), color="red", ls="--", lw=2, label=f"Median: {d.median():.1f}")
    axes[i].set_xlabel(title); axes[i].legend(fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numerical Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f"{OUT}/numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.close(); print("2/10 - Numerical distributions")

# 3. Boxplots: Good vs Bad
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
box_cols = ["loan_amnt", "dti", "fico_score", "int_rate", "annual_inc", "revol_util"]
for ax, col in zip(axes.flat, box_cols):
    sns.boxplot(data=df_acc, x="bad_loan", y=col, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_xlabel(""); ax.set_xticklabels(["Good", "Bad"])
plt.suptitle("Boxplots: Good vs Bad Loans", fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/boxplots_good_vs_bad.png", dpi=150, bbox_inches="tight")
plt.close(); print("+ Boxplots")

# 4. Categorical distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
grade_order = ["A", "B", "C", "D", "E", "F", "G"]
gc = df_acc["grade"].value_counts().reindex(grade_order)
axes[0].bar(gc.index, gc.values, color=sns.color_palette("viridis", 7))
axes[0].set_title("Loan Grade")
for i, v in enumerate(gc.values):
    axes[0].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
tc = df_acc["term"].value_counts()
axes[1].bar(tc.index, tc.values, color=sns.color_palette("mako", 2))
axes[1].set_title("Term (months)")
for i, v in enumerate(tc.values):
    axes[1].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
ho = df_acc["home_ownership"].value_counts()
axes[2].bar(ho.index, ho.values, color=sns.color_palette("Set2", len(ho)))
axes[2].set_title("Home Ownership")
for i, v in enumerate(ho.values):
    axes[2].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
pc = df_acc["purpose"].value_counts()
axes[3].barh(range(len(pc)), pc.values, color=sns.color_palette("viridis", len(pc)))
axes[3].set_yticks(range(len(pc))); axes[3].set_yticklabels(pc.index)
axes[3].set_title("Purpose"); axes[3].set_xlabel("Count")
vc = df_acc["verification_status"].value_counts()
axes[4].bar(vc.index, vc.values, color=sns.color_palette("mako", 3))
axes[4].set_title("Verification Status"); axes[4].tick_params(axis="x", rotation=15)
for i, v in enumerate(vc.values):
    axes[4].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
ac = df_acc["application_type"].value_counts()
axes[5].bar(ac.index, ac.values, color=sns.color_palette("Set2", 2))
axes[5].set_title("Application Type")
for i, v in enumerate(ac.values):
    axes[5].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/categorical_distributions.png", dpi=150, bbox_inches="tight")
plt.close(); print("3/10 - Categorical distributions")

# 5. Rejected loans
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df_rej["amount_requested"].dropna().hist(bins=50, ax=axes[0], color="salmon", edgecolor="white")
axes[0].set_title("Amount Requested (Rejected)")
df_rej["risk_score"].dropna().hist(bins=50, ax=axes[1], color="salmon", edgecolor="white")
axes[1].set_title("Risk Score (Rejected)")
df_rej["dti"].dropna().hist(bins=50, ax=axes[2], color="salmon", edgecolor="white")
axes[2].set_title("DTI (Rejected)")
plt.tight_layout(); plt.savefig(f"{OUT}/rejected_distributions.png", dpi=150, bbox_inches="tight")
plt.close(); print("4/10 - Rejected distributions")

# 6. Bivariate: features vs bad_loan
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col, title in zip(axes.flat, ["loan_amnt", "annual_inc", "dti", "fico_score", "int_rate", "open_acc"],
                          ["Loan Amount", "Annual Income", "DTI", "FICO Score", "Interest Rate", "Open Accounts"]):
    data = df_acc[[col, "bad_loan"]].dropna()
    for label, color, name in [(0, "#2ecc71", "Good"), (1, "#e74c3c", "Bad")]:
        subset = data[data["bad_loan"] == label][col]
        ax.hist(subset, bins=40, alpha=0.5, color=color, label=name, density=True)
    ax.set_xlabel(title); ax.legend()
plt.suptitle("Features by Loan Status", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f"{OUT}/bivariate_features.png", dpi=150, bbox_inches="tight")
plt.close(); print("5/10 - Bivariate features")

# 7. Grade vs Bad Rate
gb = df_acc.groupby("grade", observed=True)["bad_loan"].agg(["count", "mean"])
gb["mean"] *= 100
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(gb.index, gb["count"], color="lightblue")
ax1.set_ylabel("Total Loans")
ax2 = ax1.twinx()
ax2.plot(gb.index, gb["mean"], "ro-", lw=2, markersize=8)
ax2.set_ylabel("Bad Loan Rate (%)")
for i, (idx, row) in enumerate(gb.iterrows()):
    ax2.text(i, row["mean"] + 0.5, f"{row['mean']:.1f}%", ha="center", fontsize=9, color="red")
plt.title("Bad Loan Rate by Grade")
plt.tight_layout(); plt.savefig(f"{OUT}/grade_bad_rate.png", dpi=150, bbox_inches="tight")
plt.close(); print("6/10 - Grade vs Bad Rate")

# 8. Term vs Bad Rate
term_bad = df_acc.groupby("term")["bad_loan"].agg(["count", "mean"])
term_bad["mean"] *= 100
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(term_bad.index, term_bad["mean"], color=["#3498db", "#e74c3c"], width=0.4)
ax.set_ylabel("Bad Loan Rate (%)"); ax.set_title("Bad Rate by Term")
for bar, val in zip(bars, term_bad["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/term_bad_rate.png", dpi=150, bbox_inches="tight")
plt.close(); print("+ Term vs Bad Rate")

# 9. Purpose vs Bad Rate
purpose_bad = df_acc.groupby("purpose")["bad_loan"].agg(["count", "mean"])
purpose_bad["mean"] *= 100
purpose_bad = purpose_bad.sort_values("mean", ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(purpose_bad)), purpose_bad["mean"], color=sns.color_palette("Reds_r", len(purpose_bad)))
ax.set_xticks(range(len(purpose_bad))); ax.set_xticklabels(purpose_bad.index, rotation=45, ha="right")
ax.set_ylabel("Bad Loan Rate (%)"); ax.set_title("Bad Rate by Purpose")
for bar, val in zip(bars, purpose_bad["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, f"{val:.1f}%", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/purpose_bad_rate.png", dpi=150, bbox_inches="tight")
plt.close(); print("+ Purpose vs Bad Rate")

# 10. FICO vs Bad Rate
df_acc["fico_bin"] = pd.cut(df_acc["fico_score"],
    bins=[0, 600, 650, 700, 750, 800, 850],
    labels=["<600", "600-650", "650-700", "700-750", "750-800", "800+"])
fb = df_acc.groupby("fico_bin", observed=True)["bad_loan"].agg(["count", "mean"])
fb["mean"] *= 100
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(fb.index, fb["count"], color="lightblue")
ax1.set_ylabel("Total Loans"); ax1.set_xlabel("FICO Score")
ax2 = ax1.twinx()
ax2.plot(fb.index, fb["mean"], "ro-", lw=2, markersize=8)
ax2.set_ylabel("Bad Loan Rate (%)")
plt.title("Bad Loan Rate by FICO"); plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig(f"{OUT}/fico_bad_rate.png", dpi=150, bbox_inches="tight")
plt.close(); print("7/10 - FICO vs Bad Rate")

# 11. Hardship + Debt Settlement
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hr = df_acc.groupby("had_hardship")["bad_loan"].mean()
axes[0].bar(["No Hardship", "Hardship"], hr.values * 100, color=["#3498db", "#e74c3c"], width=0.4)
axes[0].set_ylabel("Bad Loan Rate (%)"); axes[0].set_title("Impact of Hardship")
for i, v in enumerate(hr.values):
    axes[0].text(i, v*100+1, f"{v*100:.1f}%", ha="center", fontweight="bold")
dr = df_acc.groupby("debt_settlement")["bad_loan"].mean()
axes[1].bar(["No Settle", "Settlement"], dr.values * 100, color=["#3498db", "#e74c3c"], width=0.4)
axes[1].set_ylabel("Bad Loan Rate (%)"); axes[1].set_title("Impact of Debt Settlement")
for i, v in enumerate(dr.values):
    axes[1].text(i, v*100+1, f"{v*100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/hardship_settlement.png", dpi=150, bbox_inches="tight")
plt.close(); print("+ Hardship analysis")

# 12. Correlation
num_df = df_acc.select_dtypes(include=np.number)
corr = num_df.corr()
target_corr = corr["bad_loan"].abs().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
sns.heatmap(corr[["bad_loan"]].sort_values(by="bad_loan", ascending=False),
    annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title("Correlation with Bad Loan")
top_f = target_corr.head(20).index.tolist()
top_f.remove("bad_loan")
top_f = ["bad_loan"] + top_f[:14]
sns.heatmap(num_df[top_f].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5, ax=axes[1])
axes[1].set_title("Correlation Matrix - Top 15")
plt.tight_layout(); plt.savefig(f"{OUT}/correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close(); print("9/10 - Correlation analysis")

# 13. Geographic
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ss = df_acc.groupby("addr_state")["bad_loan"].agg(["count", "mean"])
ss["mean"] *= 100; ss = ss[ss["count"] > 10].sort_values("mean", ascending=False)
ts = ss.head(15)
axes[0].barh(range(len(ts)), ts["mean"], color="coral")
axes[0].set_yticks(range(len(ts))); axes[0].set_yticklabels(ts.index)
axes[0].set_xlabel("Bad Loan Rate (%)"); axes[0].set_title("Top 15 States by Bad Rate"); axes[0].invert_yaxis()
ml = ss.sort_values("count", ascending=False).head(15)
axes[1].barh(range(len(ml)), ml["count"], color="steelblue")
axes[1].set_yticks(range(len(ml))); axes[1].set_yticklabels(ml.index)
axes[1].set_xlabel("Loan Count"); axes[1].set_title("Top 15 States by Volume"); axes[1].invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/state_analysis.png", dpi=150, bbox_inches="tight")
plt.close(); print("10/10 - Geographic analysis")

# 14. Accepted vs Rejected
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_acc["loan_amnt"].dropna(), bins=40, alpha=0.6, color="#2ecc71", label="Accepted", density=True)
axes[0].hist(df_rej["amount_requested"].dropna(), bins=40, alpha=0.6, color="#e74c3c", label="Rejected", density=True)
axes[0].set_xlabel("Amount"); axes[0].legend(); axes[0].set_title("Amount: Accepted vs Rejected")
axes[1].hist(df_acc["dti"].dropna(), bins=40, alpha=0.6, color="#2ecc71", label="Accepted", density=True)
axes[1].hist(df_rej["dti"].dropna(), bins=40, alpha=0.6, color="#e74c3c", label="Rejected", density=True)
axes[1].set_xlabel("DTI %"); axes[1].legend(); axes[1].set_title("DTI: Accepted vs Rejected")
plt.suptitle("Comparison: Accepted vs Rejected", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/accepted_vs_rejected.png", dpi=150, bbox_inches="tight")
plt.close(); print("+ Accepted vs Rejected")

# Save EDA summary
pct_bad = df_acc["bad_loan"].mean() * 100
summary = f"""EDA SUMMARY - LendingClub Loans
{'='*60}
1. TARGET: Bad Loan = {pct_bad:.1f}% (desbalanceado, requiere SMOTE)
2. TOP CORRELATIONS WITH bad_loan:
"""
for col, val in target_corr.head(10).items():
    summary += f"   {col}: {val:.4f}\\n"
summary += f"""
3. FEATURES WITH >50% MISSING:
"""
high_missing = df_acc.columns[df_acc.isnull().mean() > 0.5].tolist()
for c in high_missing:
    summary += f"   {c}: {df_acc[c].isnull().mean()*100:.1f}%\\n"
summary += f"\\nTotal: {len(high_missing)} columns\\n"
summary += f"""
4. GRADE STATS:
"""
for idx, row in gb.iterrows():
    summary += f"   {idx}: {int(row['count']):,} loans, {row['mean']:.1f}% bad rate\\n"
with open(f"{OUT}/eda_summary.txt", "w") as f:
    f.write(summary)
print(f"\\nDone! All figures in {OUT}/")
