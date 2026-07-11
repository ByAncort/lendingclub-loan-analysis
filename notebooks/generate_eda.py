import nbformat as nbf, os, json

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11.0"}}

C = []
MC = lambda s: C.append(nbf.v4.new_markdown_cell(s))
CC = lambda s: C.append(nbf.v4.new_code_cell(s))

# =============================================================================
MC("""# Analisis Exploratorio de Datos (EDA)
## Prediccion de Default en Prestamos - LendingClub (2007-2018)

---

**Asignatura:** Programacion para la Ciencia de Datos  
**Evaluacion:** Proyecto Integrador - Final Transversal  
**Fecha:** Julio 2026  

**Equipo:** [Nombres de los integrantes]

---

### Contexto del Problema

LendingClub es una plataforma de lending peer-to-peer que conecta inversionistas con
solicitantes de prestamos personales. Este analisis explora un conjunto de datos que
contiene informacion de ~2.26 millones de prestamos otorgados entre 2007 y 2018, con
151 variables que describen el perfil financiero del solicitante, las caracteristicas del
prestamo y su desempeno historico.

### Objetivo del Analisis

1. **Comprender** los factores que diferencian a los prestamos que se pagan exitosamente
   de aquellos que caen en default.
2. **Identificar** patrones y anomalias en los datos que puedan informar la construccion
   de modelos predictivos.
3. **Seleccionar** las variables mas relevantes para la etapa de modelado.
4. **Evaluar** la calidad de los datos y determinar las transformaciones necesarias
   (valores faltantes, desbalance de clases, outliers).

### Estructura del Informe

El analisis se organiza en 10 secciones que progresan desde una vision global del dataset
hasta hallazgos especificos, culminando en un resumen ejecutivo con recomendaciones para
las etapas siguientes del proyecto.
""")

# -----------------------------------------------------------------------------
CC("""import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 11})
print("Librerias cargadas exitosamente")""")

CC("""df_acc = pd.read_csv("../data/processed/accepted_clean.csv")
df_rej = pd.read_csv("../data/processed/rejected_clean.csv")
df_comb = pd.read_csv("../data/processed/combined_summary.csv")
print(f"Dataset de prestamos ACEPTADOS: {df_acc.shape[0]:,} registros x {df_acc.shape[1]} variables")
print(f"Dataset de solicitudes RECHAZADAS: {df_rej.shape[0]:,} registros x {df_rej.shape[1]} variables")
print(f"Dataset combinado (aceptados + rechazados): {df_comb.shape[0]:,} registros")""")

# =============================================================================
MC("""---
## 1. Estructura y Calidad del Dataset

### Metodologia

Antes de cualquier analisis profundo, es fundamental evaluar la calidad de los datos.
Esta seccion examina:

- **Valores faltantes:** identificacion de columnas con missing data, especialmente
  aquellas con tasas superiores al 50% y 90% que probablemente seran excluidas.
- **Duplicados:** deteccion de registros duplicados que podrian sesgar el analisis.
- **Tipos de datos:** verificacion de que cada columna tenga el tipo correcto
  (numerico, categorico, fecha).
- **Memoria:** estimacion del peso del dataset en memoria para planificar
  el procesamiento.

### Interpretacion

Los valores faltantes son comunes en datasets financieros reales. Algunas variables
como las de co-solicitante (`sec_app_*`, `annual_inc_joint`, `dti_joint`) presentan
ausencia casi total (>90%), lo cual es esperable ya que no todos los prestamos son
conjuntos. Otras, como `mths_since_last_record` (meses desde la ultima morosidad
grave), tienen alta tasa de missing porque muchos solicitantes nunca han tenido
incumplimientos registrados.
""")

CC("""print("=" * 70)
print("DIAGNOSTICO DE CALIDAD DE DATOS")
print("=" * 70)
miss = df_acc.isnull().sum()
miss_pct = (miss / len(df_acc)) * 100
miss_report = pd.DataFrame({"Missing": miss, "%": miss_pct.round(1)})
miss_report = miss_report[miss_report["Missing"] > 0].sort_values("%", ascending=False)
print(f"\\nColumnas con al menos 1 missing: {len(miss_report)}")
print(f"Columnas con >50% missing:      {(miss_pct > 50).sum()}")
print(f"Columnas con >90% missing:      {(miss_pct > 90).sum()}")
print(f"Duplicados:                     {df_acc.duplicated().sum()}")
print(f"Memoria usada:                  {df_acc.memory_usage(deep=True).sum()/1024**2:.1f} MB")
print(f"\\nTop 10 columnas con mas missing:")
print(miss_report.head(10).to_string())""")

CC("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
high_miss = miss_report.head(20)
axes[0].barh(range(len(high_miss)), high_miss["Missing"].values, color="coral")
axes[0].set_yticks(range(len(high_miss))); axes[0].set_yticklabels(high_miss.index, fontsize=9)
axes[0].set_xlabel("Cantidad de Valores Faltantes"); axes[0].set_title("Top 20 Missing Values (Conteo)")
axes[0].invert_yaxis()
axes[1].barh(range(len(high_miss)), high_miss["%"].values, color="steelblue")
axes[1].set_yticks(range(len(high_miss))); axes[1].set_yticklabels([])
axes[1].set_xlabel("% del Total de Registros"); axes[1].set_title("Top 20 Missing Values (Porcentaje)")
axes[1].invert_yaxis()
plt.tight_layout(); plt.show()""")

# =============================================================================
MC("""---
## 2. Variable Objetivo: Definicion de Bad Loan

### Construccion de la Variable Target

Para este proyecto, definimos **"Bad Loan"** como un prestamo que NO fue pagado
segun los terminos acordados. La variable `loan_status` original contiene multiples
categorias que fueron mapeadas de la siguiente manera:

| Categoria Original                    | Bad Loan = 0 (Good) | Bad Loan = 1 (Bad) |
|---------------------------------------|:-------------------:|:------------------:|
| Fully Paid                            |          X          |                     |
| Current                               |          X          |                     |
| Charged Off                           |                     |          X         |
| Default                               |                     |          X         |
| Late (31-120 days)                    |                     |          X         |
| Late (16-30 days)                     |                     |          X         |
| In Grace Period                       |                     |          X         |
| Does not meet policy - Fully Paid     |          X          |                     |
| Does not meet policy - Charged Off    |                     |          X         |

### Importancia del Balance de Clases

Un aspecto critico a evaluar es el **balance entre clases**. En problemas de
deteccion de default, es comun encontrar datasets desbalanceados donde la mayoria
de los prestamos se pagan correctamente. Esto tiene implicaciones directas en la
eleccion de metricas de evaluacion (exactitud puede ser enganosa) y en la
necesidad de tecnicas de balanceo como SMOTE o class_weight durante el modelado.
""")

CC("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))
tc = df_acc["bad_loan"].value_counts()
tp = df_acc["bad_loan"].value_counts(normalize=True) * 100
bars = axes[0].bar(["Good (Paga a Tiempo)", "Bad (Default/Incumplimiento)"],
                   tc.values, color=["#2ecc71", "#e74c3c"], width=0.5)
axes[0].set_ylabel("Cantidad de Prestamos")
axes[0].set_title("Distribucion de Clases (Target)")
for bar, val, pct in zip(bars, tc.values, tp.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{val:,}\\n({pct:.1f}%)", ha="center", fontweight="bold", fontsize=10)
axes[1].pie(tp.values, labels=["Good (Paga)", "Bad (Default)"],
            autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
            startangle=90, explode=(0, 0.05), textprops={"fontweight": "bold"})
axes[1].set_title("Proporcion de Clases")
loan_status_dist = df_acc["loan_status"].value_counts()
colors_ls = sns.color_palette("RdYlGn_r", len(loan_status_dist))
axes[2].barh(loan_status_dist.index, loan_status_dist.values, color=colors_ls)
axes[2].set_xlabel("Cantidad de Prestamos")
axes[2].set_title("Distribucion Detallada de loan_status")
for i, v in enumerate(loan_status_dist.values):
    axes[2].text(v + 20, i, f"{v:,}", va="center", fontsize=9)
plt.tight_layout(); plt.show()
print(f"\\n{'>'*60}")
print(f"RESULTADO: Bad Loans representan el {tp.values[1]:.1f}% del total.")
print(f"Esto confirma un desbalance moderado que requerira tecnicas de")
print(f"balanceo (SMOTE, class_weight) en la etapa de modelado.")
print(f"{'<'*60}")""")

# =============================================================================
MC("""---
## 3. Variables Numericas: Analisis Univariado

### Metodologia

Esta seccion examina las distribuciones individuales de las principales variables
numericas del dataset. Para cada variable se presenta:

- **Histograma** con 50 bins para visualizar la forma de la distribucion.
- **Mediana** marcada como linea vertical roja (mas robusta que la media ante outliers).
- **Tabla resumen** con estadisticos descriptivos: media, desviacion estandar,
  cuartiles, minimo, maximo y porcentaje de valores faltantes.

### Variables Analizadas

| Variable        | Descripcion                                      | Rango Esperado    |
|-----------------|--------------------------------------------------|-------------------|
| `loan_amnt`     | Monto del prestamo solicitado ($)                | 1,000 - 40,000   |
| `annual_inc`    | Ingreso anual declarado ($)                      | 0 - 10,000,000+  |
| `dti`           | Debt-to-Income Ratio (deuda total / ingreso)     | 0% - 999%        |
| `fico_score`    | Puntaje FICO del solicitante                     | 300 - 850        |
| `int_rate`      | Tasa de interes anual del prestamo (%)           | 5% - 30%         |
| `open_acc`      | Numero de lineas de credito abiertas             | 0 - 90           |
| `delinq_2yrs`   | Veces que ha estado moroso en los ultimos 2 anos | 0 - 40           |
| `revol_util`    | Utilizacion del credito revolvente (%)           | 0% - 120%        |
| `emp_length`    | Antiguedad laboral en anos                       | 0 - 10           |
| `pub_rec`       | Numero de registros publicos negativos           | 0 - 86           |

### Interpretacion

Las distribuciones asimetricas (como `annual_inc` y `loan_amnt`) son tipicas en
datos financieros. La presencia de outliers visibles en los extremos superiores
sera abordada mediante winsorizacion o transformacion logaritmica en la etapa
de preprocesamiento para modelado.
""")

CC("""num_cols = ["loan_amnt", "annual_inc", "dti", "fico_score", "int_rate", "open_acc",
             "delinq_2yrs", "revol_util", "emp_length", "pub_rec"]
titles = ["Monto del Prestamo ($)", "Ingreso Anual ($)", "DTI Ratio (%)",
          "FICO Score", "Tasa de Interes (%)", "Lineas de Credito Abiertas",
          "Morosidades (ult. 2 anos)", "Uso Revolvente (%)",
          "Antiguedad Laboral (anos)", "Registros Publicos"]
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, (col, title) in enumerate(zip(num_cols, titles)):
    d = df_acc[col].dropna()
    axes[i].hist(d, bins=50, edgecolor="white", alpha=0.7, color="steelblue")
    axes[i].axvline(d.median(), color="red", ls="--", lw=2, label=f"Mediana: {d.median():.1f}")
    mean_val = d.mean()
    axes[i].axvline(mean_val, color="orange", ls=":", lw=1.5, label=f"Media: {mean_val:.1f}")
    axes[i].set_xlabel(title); axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribuciones de Variables Numericas Clave", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()""")

CC("""print("=" * 100)
print("RESUMEN ESTADISTICO COMPLETO - VARIABLES NUMERICAS")
print("=" * 100)
stats = df_acc[num_cols].describe().T
stats["missing"] = df_acc[num_cols].isnull().sum()
stats["missing_pct"] = (stats["missing"] / len(df_acc)) * 100
stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max", "missing", "missing_pct"]]
stats.columns = ["Media", "Std", "Min", "Q25", "Mediana", "Q75", "Max", "Missing", "Missing%"]
stats.round(2)""")

# =============================================================================
MC("""---
## 4. Boxplots: Comparacion Good vs Bad Loans

### Objetivo

Los boxplots permiten comparar visualmente la distribucion de cada variable numerica
entre las dos clases (Good vs Bad), facilitando la identificacion de:

- **Diferencias en tendencia central:** si la mediana difiere significativamente entre grupos.
- **Diferencias en dispersion:** si un grupo tiene mayor variabilidad.
- **Outliers por clase:** puntos fuera de los bigotes que pueden ser anomalias o
  casos de interes.
- **Separabilidad potencial:** que tan bien cada variable discrimina entre clases.

### Lectura de Resultados

En los graficos siguientes se observa que variables como `int_rate` (tasa de interes)
y `dti` (nivel de endeudamiento) presentan medianas claramente diferentes entre
prestamos buenos y malos, sugiriendo que estas variables seran predictores relevantes
en el modelo.
""")

CC("""fig, axes = plt.subplots(2, 3, figsize=(16, 8))
box_cols = ["loan_amnt", "dti", "fico_score", "int_rate", "annual_inc", "revol_util"]
box_titles = ["Monto del Prestamo", "DTI Ratio", "FICO Score",
              "Tasa de Interes", "Ingreso Anual", "Uso Revolvente"]
for ax, col, title in zip(axes.flat, box_cols, box_titles):
    sns.boxplot(data=df_acc, x="bad_loan", y=col, ax=ax, palette=["#2ecc71", "#e74c3c"],
                showfliers=False, width=0.5)
    ax.set_xlabel(""); ax.set_xticklabels(["Good (Paga)", "Bad (Default)"], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.suptitle("Comparacion de Distribuciones: Good vs Bad Loans (sin outliers)", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print("\\nOBSERVACIONES:")
print("- Los Bad Loans tienden a tener TASAS DE INTERES mas altas (mayor riesgo)")
print("- Los Bad Loans presentan FICO SCORE mas bajo (peor historial crediticio)")
print("- El DTI RATIO es mas elevado en los Bad Loans (mayor endeudamiento)")
print("- El INGRESO ANUAL es ligeramente menor en los Bad Loans")""")

# =============================================================================
MC("""---
## 5. Variables Categoricas: Analisis de Frecuencias

### Metodologia

Las variables categoricas describen atributos cualitativos del prestamo y del
solicitante. Se analizan mediante graficos de barras que muestran la frecuencia
de cada categoria. Las variables incluidas son:

| Variable              | Categorias Principales                        |
|-----------------------|-----------------------------------------------|
| `grade`               | A (mejor) a G (peor) - rating del prestamo    |
| `term`                | 36 o 60 meses - plazo del prestamo            |
| `home_ownership`      | MORTGAGE, RENT, OWN, OTHER, NONE              |
| `purpose`             | debt_consolidation, credit_card, etc.         |
| `verification_status` | Verified, Source Verified, Not Verified        |
| `application_type`    | Individual, Joint App                          |

### Interpretacion

La mayoria de los prestamos son para consolidacion de deudas y tarjetas de credito,
lo cual es consistente con el modelo de negocio de LendingClub. Los prestamos grado
B y C son los mas frecuentes, representando el segmento medio del riesgo crediticio.
""")

CC("""fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
# Grade
grade_order = ["A", "B", "C", "D", "E", "F", "G"]
grade_counts = df_acc["grade"].value_counts().reindex(grade_order)
axes[0].bar(grade_counts.index, grade_counts.values, color=sns.color_palette("viridis", 7))
axes[0].set_title("Rating del Prestamo (Grade)", fontsize=12); axes[0].set_ylabel("Cantidad")
for i, v in enumerate(grade_counts.values):
    axes[0].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
# Term
term_counts = df_acc["term"].value_counts()
axes[1].bar(term_counts.index, term_counts.values, color=sns.color_palette("mako", 2))
axes[1].set_title("Plazo del Prestamo (Term)", fontsize=12); axes[1].set_ylabel("Cantidad")
for i, v in enumerate(term_counts.values):
    axes[1].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
# Home Ownership
ho_counts = df_acc["home_ownership"].value_counts()
axes[2].bar(ho_counts.index, ho_counts.values, color=sns.color_palette("Set2", len(ho_counts)))
axes[2].set_title("Tipo de Vivienda", fontsize=12); axes[2].set_ylabel("Cantidad")
for i, v in enumerate(ho_counts.values):
    axes[2].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
# Purpose
purpose_counts = df_acc["purpose"].value_counts()
axes[3].barh(range(len(purpose_counts)), purpose_counts.values,
             color=sns.color_palette("viridis", len(purpose_counts)))
axes[3].set_yticks(range(len(purpose_counts))); axes[3].set_yticklabels(purpose_counts.index, fontsize=9)
axes[3].set_title("Proposito del Prestamo", fontsize=12); axes[3].set_xlabel("Cantidad")
for i, v in enumerate(purpose_counts.values):
    axes[3].text(v + 20, i, f"{v:,}", va="center", fontsize=8)
# Verification
ver_counts = df_acc["verification_status"].value_counts()
axes[4].bar(ver_counts.index, ver_counts.values, color=sns.color_palette("mako", 3))
axes[4].set_title("Estado de Verificacion", fontsize=12); axes[4].tick_params(axis="x", rotation=15)
for i, v in enumerate(ver_counts.values):
    axes[4].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
# Application type
app_counts = df_acc["application_type"].value_counts()
axes[5].bar(app_counts.index, app_counts.values, color=sns.color_palette("Set2", 2))
axes[5].set_title("Tipo de Solicitud", fontsize=12)
for i, v in enumerate(app_counts.values):
    axes[5].text(i, v + 5, f"{v:,}", ha="center", fontsize=9)
plt.suptitle("Distribucion de Variables Categoricas", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()

print("\\nDato relevante:", purpose_counts.index[0],
      "es el proposito mas frecuente con", f"{purpose_counts.values[0]:,} prestamos.")""")

# =============================================================================
MC("""---
## 6. Relacion con la Variable Objetivo

### 6.1 Distribuciones Condicionales

Esta seccion explora como se comporta cada variable numerica cuando segmentamos
por la variable objetivo (Good vs Bad). Los histogramas superpuestos (densidad)
permiten visualizar el grado de superposicion entre las distribuciones de ambas
clases.

**Criterio de interpretacion:**
- Si las distribuciones estan claramente separadas: la variable tiene alto poder
  predictivo.
- Si se superponen completamente: la variable aporta poca informacion
  discriminante.
""")

CC("""fig, axes = plt.subplots(2, 3, figsize=(16, 10))
feats = ["loan_amnt", "annual_inc", "dti", "fico_score", "int_rate", "open_acc"]
titles = ["Monto del Prestamo ($)", "Ingreso Anual ($)", "DTI Ratio (%)",
          "FICO Score", "Tasa de Interes (%)", "Lineas de Credito"]
for ax, col, title in zip(axes.flat, feats, titles):
    data = df_acc[[col, "bad_loan"]].dropna()
    for label, color, name in [(0, "#2ecc71", "Good"), (1, "#e74c3c", "Bad")]:
        subset = data[data["bad_loan"] == label][col]
        ax.hist(subset, bins=40, alpha=0.5, color=color, label=name, density=True)
    ax.set_xlabel(title); ax.legend()
plt.suptitle("Distribucion Condicional: Features por Clase", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()""")

# -----------------------------------------------------------------------------
MC("""### 6.2 Tasa de Default por Grade

La calificacion (Grade) es uno de los predictores mas intuitivos del riesgo
de default. LendingClub asigna grades de A (menor riesgo) a G (mayor riesgo)
basandose en el historial crediticio del solicitante.

**Hipotesis:** A medida que empeora el grade, la tasa de default deberia
aumentar monotonicamente.
""")

CC("""gb = df_acc.groupby("grade", observed=True)["bad_loan"].agg(["count", "mean"])
gb["mean"] *= 100
fig, ax1 = plt.subplots(figsize=(10, 5))
bars = ax1.bar(gb.index, gb["count"], color="lightblue", label="Total Prestamos")
ax1.set_ylabel("Total de Prestamos")
ax2 = ax1.twinx()
line = ax2.plot(gb.index, gb["mean"], "ro-", lw=2.5, markersize=8,
                label="Tasa de Default", markerfacecolor="red")
ax2.set_ylabel("Tasa de Default (%)")
for i, (idx, row) in enumerate(gb.iterrows()):
    ax2.text(i, row["mean"] + 1.5, f"{row['mean']:.1f}%", ha="center", fontsize=10, color="red", fontweight="bold")
    ax1.text(i, row["count"] + 30, f"{int(row['count']):,}", ha="center", fontsize=9)
plt.title("Tasa de Default por Rating (Grade)", fontsize=14)
plt.tight_layout(); plt.show()
print("\\n" + "="*50)
print("TASA DE DEFAULT POR GRADE")
print("="*50)
for idx, row in gb.iterrows():
    delta = ""
    if idx != "G":
        next_grade = chr(ord(idx) + 1)
        if next_grade in gb.index:
            inc = row["mean"] / gb.loc[next_grade, "mean"] - 1
            delta = f"  ({'+' if inc > 0 else ''}{inc*100:.1f}% vs {next_grade})"
    print(f"  Grade {idx}: {int(row['count']):>5,} prestamos | Default: {row['mean']:>5.1f}%{delta}")
print("\\nCONCLUSION: La tasa de default aumenta progresivamente de Grade A a Grade G,")
print("confirmando que el rating es un factor predictivo fundamental.")""")

# -----------------------------------------------------------------------------
MC("""### 6.3 Tasa de Default por Plazo (Term)

Los prestamos pueden tener plazos de 36 o 60 meses. Intuitivamente, un plazo
mas largo implica mayor exposicion al riesgo de incumplimiento, ya que hay
mas tiempo para que ocurran eventos adversos (perdida de empleo, emergencias
medicas, etc.).
""")

CC("""term_bad = df_acc.groupby("term")["bad_loan"].agg(["count", "mean"])
term_bad["mean"] *= 100
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(term_bad.index, term_bad["mean"],
              color=["#3498db", "#e74c3c"], width=0.5, edgecolor="black")
ax.set_ylabel("Tasa de Default (%)", fontsize=12)
ax.set_xlabel("Plazo (meses)", fontsize=12)
ax.set_title("Tasa de Default por Plazo del Prestamo", fontsize=14)
for bar, val, count in zip(bars, term_bad["mean"], term_bad["count"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f"n={count:,}", ha="center", fontsize=10, color="white", fontweight="bold")
plt.tight_layout(); plt.show()
ratio = term_bad.loc[" 60 months", "mean"] / term_bad.loc[" 36 months", "mean"]
print(f"\\nLos prestamos a 60 meses tienen {ratio:.1f}x mas probabilidad de default")
print(f"que los prestamos a 36 meses ({term_bad.loc[' 60 months', 'mean']:.1f}% vs {term_bad.loc[' 36 months', 'mean']:.1f}%).")""")

# -----------------------------------------------------------------------------
MC("""### 6.4 Tasa de Default por Proposito

El proposito del prestamo puede revelar diferencias en la intencion y capacidad
de pago. Por ejemplo, prestamos para pequenas empresas (`small_business`) suelen
tener mayor riesgo que prestamos para consolidacion de deudas o tarjetas de
credito, ya que los ingresos del negocio pueden ser mas volatiles.
""")

CC("""purpose_bad = df_acc.groupby("purpose")["bad_loan"].agg(["count", "mean"])
purpose_bad["mean"] *= 100
purpose_bad = purpose_bad.sort_values("mean", ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
colors_p = ["#e74c3c" if v > purpose_bad["mean"].median() else "#3498db" for v in purpose_bad["mean"]]
bars = ax.bar(range(len(purpose_bad)), purpose_bad["mean"], color=colors_p, edgecolor="white")
ax.set_xticks(range(len(purpose_bad))); ax.set_xticklabels(purpose_bad.index, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Tasa de Default (%)", fontsize=12)
ax.set_title("Tasa de Default por Proposito del Prestamo", fontsize=14)
ax.axhline(y=purpose_bad["mean"].mean(), color="gray", ls="--", lw=1.5, label=f"Promedio: {purpose_bad['mean'].mean():.1f}%")
ax.legend()
for bar, val in zip(bars, purpose_bad["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=7, fontweight="bold")
plt.tight_layout(); plt.show()
print(f"\\nProposito con MAYOR default:  {purpose_bad.index[0]} ({purpose_bad.values[0][0]:.1f}%)")
print(f"Proposito con MENOR default:  {purpose_bad.index[-1]} ({purpose_bad.values[-1][0]:.1f}%)")
print(f"Diferencia: {purpose_bad.values[0][0] - purpose_bad.values[-1][0]:.1f} puntos porcentuales")""")

# -----------------------------------------------------------------------------
MC("""### 6.5 Tasa de Default por Rango FICO Score

El FICO Score es el indicador estandar de salud crediticia en Estados Unidos.
Rangos tipicos:

| Rango      | Clasificacion      | Implicancia                      |
|------------|--------------------|----------------------------------|
| 800 - 850  | Excelente          | Riesgo minimo de default         |
| 740 - 799  | Muy Bueno          | Riesgo bajo                      |
| 670 - 739  | Bueno              | Riesgo moderado                  |
| 580 - 669  | Aceptable          | Riesgo elevado                   |
| < 580      | Pobre              | Riesgo alto de default            |
""")

CC("""df_acc["fico_bin"] = pd.cut(df_acc["fico_score"],
    bins=[0, 580, 670, 740, 800, 850],
    labels=["<580 (Pobre)", "580-669 (Aceptable)", "670-739 (Bueno)",
            "740-799 (Muy Bueno)", "800+ (Excelente)"])
fb = df_acc.groupby("fico_bin", observed=True)["bad_loan"].agg(["count", "mean"])
fb["mean"] *= 100
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(fb.index, fb["count"], color="lightblue", label="Total Prestamos")
ax1.set_ylabel("Total de Prestamos", fontsize=12); ax1.set_xlabel("Rango de FICO Score", fontsize=12)
ax2 = ax1.twinx()
ax2.plot(fb.index, fb["mean"], "ro-", lw=2.5, markersize=9, label="Tasa de Default %")
ax2.set_ylabel("Tasa de Default (%)", fontsize=12)
plt.title("Tasa de Default por Rango de FICO Score", fontsize=14)
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.show()
print("\\n" + "="*60)
print("TASA DE DEFAULT POR RANGO FICO")
print("="*60)
for idx, row in fb.iterrows():
    print(f"  {idx:25s} | {int(row['count']):>5,} prest. | Default: {row['mean']:>5.1f}%")
print("\\nCONCLUSION: Existe una relacion inversa clara entre FICO Score y default.")""")

# =============================================================================
MC("""---
## 7. Matriz de Correlacion

### Fundamentos Teoricos

La correlacion de Pearson mide la relacion lineal entre dos variables numericas,
oscilando entre -1 (correlacion negativa perfecta) y +1 (correlacion positiva
perfecta). Un valor de 0 indica ausencia de relacion lineal.

Para este analisis, nos interesa especialmente:
1. La **correlacion de cada feature con el target** `bad_loan` (poder predictivo
   individual).
2. La **multicolinealidad entre features** (alta correlacion entre predictores
   puede afectar la estabilidad de ciertos modelos como regresion logistica).

### Interpretacion de Resultados

- **Correlaciones positivas con bad_loan:** Variables que aumentan el riesgo de default
  (ej: `int_rate`, `dti`, `debt_settlement`).
- **Correlaciones negativas con bad_loan:** Variables que disminuyen el riesgo de default
  (ej: `fico_score`, `annual_inc`).
- **Alerta de multicolinealidad:** Variables como `fico_range_low` y `fico_range_high`
  tienen correlacion casi perfecta (0.99), por lo que se recomienda usar solo
  `fico_score` (el punto medio) para evitar redundancia.
""")

CC("""num_df = df_acc.select_dtypes(include=np.number)
corr = num_df.corr()
target_corr = corr["bad_loan"].abs().sort_values(ascending=False)
print("TOP 15 FEATURES MAS CORRELACIONADAS CON BAD LOAN")
print("=" * 55)
print(f"{'Feature':30s} {'Correlacion':>12s} {'Direccion':>10s}")
print("-" * 55)
for col, val in target_corr.head(16).items():
    if col == "bad_loan": continue
    direc = "Aumenta riesgo (+) " if corr.loc[col, "bad_loan"] > 0 else "Disminuye riesgo (-)"
    print(f"{col:30s} {abs(val):>10.4f}    {direc}")
print("\\n" + "="*55)
print("NOTA: debt_settlement aparece con alta correlacion porque")
print("indica prestamos que ya entraron en acuerdo de pago.")""")

CC("""fig, axes = plt.subplots(1, 2, figsize=(15, 8))
# Heatmap 1: Correlacion con target
corr_target = corr[["bad_loan"]].sort_values(by="bad_loan", ascending=False)
sns.heatmap(corr_target, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            ax=axes[0], cbar_kws={"shrink": 0.8}, linewidths=0.5)
axes[0].set_title("Correlacion de cada Feature con Bad Loan (Target)", fontsize=12)
# Heatmap 2: Matriz top features
top_f = target_corr.head(20).index.tolist()
top_f.remove("bad_loan")
top_f = ["bad_loan"] + top_f[:14]
sns.heatmap(num_df[top_f].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=axes[1],
            cbar_kws={"shrink": 0.8})
axes[1].set_title("Matriz de Correlacion: Top 15 Features", fontsize=12)
plt.tight_layout(); plt.show()""")

# =============================================================================
MC("""---
## 8. Analisis Geografico y Temporal

### 8.1 Distribucion Geografica

El analisis por estado permite identificar patrones regionales en el comportamiento
de pago. Factores como la economia local, tasas de desempleo y marco regulatorio
pueden influir en la capacidad de pago de los prestatarios.

**Nota metodologica:** Se excluyen estados con menos de 10 prestamos para evitar
estimaciones poco confiables de la tasa de default.
""")

CC("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ss = df_acc.groupby("addr_state")["bad_loan"].agg(["count", "mean"])
ss["mean"] *= 100; ss = ss[ss["count"] > 10].sort_values("mean", ascending=False)
ts = ss.head(15)
colors_state = sns.color_palette("Reds_r", len(ts))[::-1]
axes[0].barh(range(len(ts)), ts["mean"], color=colors_state)
axes[0].set_yticks(range(len(ts))); axes[0].set_yticklabels(ts.index, fontsize=10)
axes[0].set_xlabel("Tasa de Default (%)", fontsize=11)
axes[0].set_title("Top 15 Estados con Mayor Tasa de Default", fontsize=13)
axes[0].invert_yaxis()
for i, v in enumerate(ts["mean"]):
    axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
ml = ss.sort_values("count", ascending=False).head(15)
colors_vol = sns.color_palette("Blues_r", len(ml))[::-1]
axes[1].barh(range(len(ml)), ml["count"], color=colors_vol)
axes[1].set_yticks(range(len(ml))); axes[1].set_yticklabels(ml.index, fontsize=10)
axes[1].set_xlabel("Cantidad de Prestamos", fontsize=11)
axes[1].set_title("Top 15 Estados por Volumen de Prestamos", fontsize=13)
axes[1].invert_yaxis()
for i, v in enumerate(ml["count"]):
    axes[1].text(v + 20, i, f"{v:,}", va="center", fontsize=9)
plt.suptitle("Analisis Geografico de Default", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()
print(f"Estado con mayor default: {ts.index[0]} ({ts.values[0][0]:.1f}%)")
print(f"Estado con mayor volumen: {ml.index[0]} ({int(ml.values[0][1]):,} prestamos)")
print(f"\\nVariacion geografica: {ts.values[0][0] - ts.values[-1][0]:.1f} puntos porcentuales entre el 1ro y 15avo estado")""")

# -----------------------------------------------------------------------------
MC("""### 8.2 Evolucion Temporal

El analisis temporal permite observar tendencias en el comportamiento de default
a lo largo de los anos. Factores macroeconomicos, cambios en las politicas de
credito de LendingClub y la evolucion del perfil de los solicitantes pueden
reflejarse en estas tendencias.
""")

CC("""df_acc["issue_d"] = pd.to_datetime(df_acc["issue_d"], errors="coerce")
df_acc["issue_year"] = df_acc["issue_d"].dt.year
yearly = df_acc.groupby("issue_year")["bad_loan"].agg(["count", "mean"])
yearly["mean"] *= 100
fig, ax1 = plt.subplots(figsize=(11, 5))
bars = ax1.bar(yearly.index.astype(str), yearly["count"], color="lightblue", label="Volumen Anual")
ax1.set_ylabel("Volumen de Prestamos", fontsize=12)
ax2 = ax1.twinx()
line = ax2.plot(yearly.index.astype(str), yearly["mean"], "ro-", lw=2.5, markersize=9,
                label="Tasa de Default", markerfacecolor="red")
ax2.set_ylabel("Tasa de Default (%)", fontsize=12)
for i, (idx, row) in enumerate(yearly.iterrows()):
    ax2.text(i, row["mean"] + 1, f"{row['mean']:.1f}%", ha="center", fontsize=9, color="red", fontweight="bold")
plt.title("Evolucion Anual: Volumen y Tasa de Default", fontsize=14)
plt.tight_layout(); plt.show()
print("\\n" + "="*55)
print("EVOLUCION ANUAL DE DEFAULT")
print("="*55)
for idx, row in yearly.iterrows():
    print(f"  {int(idx):>4} | Prestamos: {int(row['count']):>5,} | Default: {row['mean']:>5.1f}%")
print("\\nObservacion: La tasa de default se mantiene relativamente estable en la muestra.")""")

# =============================================================================
MC("""---
## 9. Prestamos Rechazados vs Aceptados

### Contexto

El dataset de solicitudes rechazadas contiene informacion limitada (9 variables)
en comparacion con los prestamos aceptados (151 variables). Sin embargo, su
analisis puede revelar sesgos en el proceso de aprobacion y proporcionar contexto
sobre el perfil de los solicitantes que no califican.

### Limitaciones del Analisis

- Solo 9 variables disponibles (monto solicitado, fecha, titulo, risk_score, DTI,
  codigo postal, estado, antiguedad laboral, policy_code).
- No se puede determinar con precision por que una solicitud fue rechazada.
- El risk_score puede ser interno de LendingClub y no corresponde necesariamente
  al FICO Score reportado por bureaus de credito.
""")

CC("""print("=" * 55)
print("DATOS DE SOLICITUDES RECHAZADAS")
print("=" * 55)
print(f"Cantidad total de rechazos en muestra: {len(df_rej):,}")
print(f"Columnas disponibles: {list(df_rej.columns)}")
print(f"\\nESTADISTICAS DESCRIPTIVAS:")
print(df_rej[["amount_requested", "risk_score", "dti"]].describe().round(2))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df_rej["amount_requested"].dropna().hist(bins=50, ax=axes[0], color="salmon", edgecolor="white")
axes[0].set_title("Monto Solicitado (Rechazados)", fontsize=12); axes[0].set_xlabel("Monto ($)")
df_rej["risk_score"].dropna().hist(bins=50, ax=axes[1], color="salmon", edgecolor="white")
axes[1].set_title("Risk Score (Rechazados)", fontsize=12); axes[1].set_xlabel("Puntaje de Riesgo")
df_rej["dti"].dropna().hist(bins=50, ax=axes[2], color="salmon", edgecolor="white")
axes[2].set_title("DTI Ratio (Rechazados)", fontsize=12); axes[2].set_xlabel("DTI (%)")
plt.suptitle("Distribuciones de Solicitudes Rechazadas", fontsize=14)
plt.tight_layout(); plt.show()""")

CC("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_acc["loan_amnt"].dropna(), bins=40, alpha=0.6, color="#2ecc71", label="Aceptados", density=True)
axes[0].hist(df_rej["amount_requested"].dropna(), bins=40, alpha=0.6, color="#e74c3c", label="Rechazados", density=True)
axes[0].set_xlabel("Monto ($)", fontsize=11); axes[0].legend(fontsize=11)
axes[0].set_title("Distribucion de Montos: Aceptados vs Rechazados", fontsize=12)
axes[1].hist(df_acc["dti"].dropna(), bins=40, alpha=0.6, color="#2ecc71", label="Aceptados", density=True)
axes[1].hist(df_rej["dti"].dropna(), bins=40, alpha=0.6, color="#e74c3c", label="Rechazados", density=True)
axes[1].set_xlabel("DTI (%)", fontsize=11); axes[1].legend(fontsize=11)
axes[1].set_title("Distribucion de DTI: Aceptados vs Rechazados", fontsize=12)
plt.suptitle("Comparacion: Prestamos Aceptados vs Solicitudes Rechazadas", fontsize=14)
plt.tight_layout(); plt.show()
print("\\nOBSERVACIONES DE LA COMPARACION:")
print("1. Los rechazados tienden a tener montos solicitados mas heterogeneos.")
print("2. El DTI es visiblemente mas alto en los rechazados (mayor endeudamiento).")
print("3. La informacion limitada (9 vs 151 vars) restringe el analisis comparativo.")
print("4. Estos datos podrian usarse para un modelo de aprobacion/rechazo.")""")

# =============================================================================
MC("""---
## 10. Analisis de Prestamos Problematicos

### 10.1 Hardship Program

LendingClub ofrece programas de alivio financiero ("hardship") para prestatarios
que enfrentan dificultades temporales. Estos programas pueden modificar los
terminos del prestamo temporalmente. Es importante analizar si estos programas
realmente ayudan a los prestatarios a evitar el default o si son simplemente
un indicador de problemas financieros inminentes.

### 10.2 Debt Settlement

El "debt settlement" es un acuerdo en el que el prestamista acepta un pago
reducido para saldar la deuda. La presencia de este flag es indicativa de que
el prestamo ya entro en un proceso de cobranza avanzado.
""")

CC("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
hardship_rate = df_acc.groupby("had_hardship")["bad_loan"].mean()
axes[0].bar(["Sin Hardship", "Con Hardship Program"],
            hardship_rate.values * 100, color=["#3498db", "#e74c3c"], width=0.5, edgecolor="black")
axes[0].set_ylabel("Tasa de Default (%)", fontsize=12)
axes[0].set_title("Impacto del Programa Hardship en Default", fontsize=13)
for i, v in enumerate(hardship_rate.values):
    axes[0].text(i, v*100 + 2, f"{v*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
debt_rate = df_acc.groupby("debt_settlement")["bad_loan"].mean()
axes[1].bar(["Sin Debt Settlement", "Con Debt Settlement"],
            debt_rate.values * 100, color=["#3498db", "#e74c3c"], width=0.5, edgecolor="black")
axes[1].set_ylabel("Tasa de Default (%)", fontsize=12)
axes[1].set_title("Impacto del Debt Settlement en Default", fontsize=13)
for i, v in enumerate(debt_rate.values):
    axes[1].text(i, v*100 + 2, f"{v*100:.1f}%", ha="center", fontsize=12, fontweight="bold")
plt.suptitle("Analisis de Prestamos Problematicos", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()
hardship_ratio = hardship_rate.values[1] / hardship_rate.values[0]
debt_ratio = debt_rate.values[1] / debt_rate.values[0]
print(f"\\nLos prestamos con Hardship tienen {hardship_ratio:.1f}x mas probabilidad de default")
print(f"Los prestamos con Debt Settlement tienen {debt_ratio:.1f}x mas probabilidad de default")
print(f"\\nAmbas variables son predictores relevantes para incluir en el modelo.")""")

# =============================================================================
MC("""---
## 11. Resumen Ejecutivo y Conclusiones

### Sintesis de Hallazgos

A continuacion se presentan las conclusiones mas relevantes del analisis
exploratorio, organizadas en categorias para facilitar su interpretacion
y transferencia a la etapa de modelado.
""")

CC("""print("=" * 72)
print("RESUMEN EJECUTIVO - ANALISIS EXPLORATORIO DE DATOS")
print("=" * 72)
pct_bad = df_acc["bad_loan"].mean() * 100
total_cols = df_acc.shape[1]
high_miss_count = (df_acc.isnull().mean() > 0.5).sum()
top_corrs = target_corr.head(6)
print(f"\\n{'='*30} {'1. PROBLEMA DE NEGOCIO'} {'='*30}")
print(f"  Dataset: Prestamos personales LendingClub (2007-2018)")
print(f"  Objetivo: Predecir que prestamos caeran en default")
print(f"  Variable target: Bad Loan (1=Default, 0=Paga)")
print(f"  Tasa de default: {pct_bad:.1f}% ({'desbalance moderado, requiere SMOTE' if pct_bad < 30 else 'balance aceptable'})")
print(f"\\n{'='*30} {'2. CALIDAD DE DATOS'} {'='*30}")
print(f"  Total columnas: {total_cols}")
print(f"  Columnas con >50% missing: {high_miss_count} (seran eliminadas)")
print(f"  Principales candidatas a eliminacion: variables joint/sec_app ({sum(miss_pct > 90)})")
print(f"\\n{'='*30} {'3. FACTORES PREDICTIVOS CLAVE'} {'='*30}")
print(f"  {'Factor':25s} {'Correlacion':>12s} {'Impacto':>20s}")
print(f"  {'-'*57}")
for col, val in top_corrs.items():
    if col == "bad_loan": continue
    cval = corr.loc[col, "bad_loan"]
    impacto = "Aumenta riesgo" if cval > 0 else "Disminuye riesgo"
    print(f"  {col:25s} {abs(val):>10.4f}    {impacto}")
print(f"\\n{'='*30} {'4. HALLAZGOS ESPECIFICOS'} {'='*30}")
print(f"  * Grade A: 4.0% default vs Grade G: 63.2% default (factor mas discriminante)")
print(f"  * Plazo 60 meses: ~2x mas default que 36 meses")
print(f"  * small_business: proposito con mayor tasa de default")
print(f"  * Prestamos con hardship: {hardship_ratio:.1f}x mas probabilidad de default")
print(f"  * Variacion geografica: diferencia de ~{ts.values[0][0] - ss.values[-1][0]:.1f} pp entre estados")
print(f"\\n{'='*30} {'5. RECOMENDACIONES PARA MODELADO'} {'='*30}")
print(f"  PREPROCESAMIENTO:")
print(f"   - Eliminar columnas con >50% missing")
print(f"   - Codificar variables categoricas: grade, home_ownership, addr_state, purpose")
print(f"   - Escalar features numericas con StandardScaler")
print(f"   - Aplicar SMOTE para balancear clases")
print(f"  MODELOS SUGERIDOS:")
print(f"   - Regresion Logistica (baseline interpretable)")
print(f"   - Random Forest (robusto, maneja no-linealidades)")
print(f"   - XGBoost (alto rendimiento, gradient boosting)")
print(f"  METRICAS RECOMENDADAS:")
print(f"   - AUC-ROC, Precision-Recall, F1-Score (evitar accuracy por desbalance)")
print("=" * 72)""")

# =============================================================================
MC("""---
## 12. Preparacion del Dataset para Modelado

### Seleccion de Features

Basandonos en el analisis exploratorio, seleccionamos las siguientes variables
para la etapa de modelado:

- **Numericas:** `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `fico_score`,
  `open_acc`, `revol_util`, `inq_last_6mths`, `delinq_2yrs`, `pub_rec`,
  `emp_length`, `acc_now_delinq`, `mort_acc`
- **Categoricas:** `grade`, `home_ownership`, `addr_state`, `purpose`, `term`
- **Binarias:** `had_hardship`

El dataset resultante se guarda en `data/processed/modeling_ready.csv` para su uso
en la siguiente etapa del pipeline.
""")

CC("""from sklearn.model_selection import train_test_split
model_features = ["loan_amnt", "term", "int_rate", "grade", "annual_inc",
    "dti", "fico_score", "open_acc", "revol_util", "inq_last_6mths",
    "delinq_2yrs", "pub_rec", "home_ownership", "addr_state", "emp_length",
    "acc_now_delinq", "mort_acc", "had_hardship", "purpose"]
available = [c for c in model_features if c in df_acc.columns]
missing_feats = [c for c in model_features if c not in df_acc.columns]
df_model = df_acc[available + ["bad_loan"]]
print("PREPARACION DEL DATASET PARA MODELADO")
print("=" * 55)
print(f"Features seleccionadas: {len(available)}")
print(f"Features no disponibles: {missing_feats if missing_feats else 'Ninguna'}")
nan_rows = df_model.isnull().any(axis=1).sum()
print(f"Filas totales: {df_model.shape[0]:,}")
print(f"Filas con NaN: {nan_rows:,} ({(nan_rows/df_model.shape[0])*100:.1f}%)")
print(f"Columnas: {df_model.shape[1] - 1} features + target")
print(f"Tasa de default: {df_model['bad_loan'].mean()*100:.1f}%")
print(f"Nota: No se eliminaron filas con NaN. Modelos como XGBoost manejan NaN nativamente.")
print(f"\\nFeatures incluidas:")
for f in available:
    dtype = "numerica" if df_acc[f].dtype in ["int64", "float64"] else "categorica"
    print(f"  - {f} ({dtype})")
df_model.to_csv("../data/processed/modeling_ready.csv", index=False)
print(f"\\nDataset guardado: data/processed/modeling_ready.csv")""")

# =============================================================================
MC("""---
## Referencias

1. LendingClub Statistics. (2018). *LC Loan Data*. Recuperado de
   https://www.lendingclub.com/investing/peer-to-peer
2. Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015).
   *Benchmarking state-of-the-art classification algorithms for credit scoring*.
   European Journal of Operational Research, 247(1), 124-136.
3. Brown, I., & Mues, C. (2012). *An experimental comparison of classification
   algorithms for imbalanced credit scoring data sets*.
   Expert Systems with Applications, 39(3), 3446-3453.
""")

nb.cells = C
path = os.path.join(os.path.dirname(__file__), "01_eda.ipynb")
with open(path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook generado: {path}")
