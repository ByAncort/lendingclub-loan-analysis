# LendingClub Risk Intelligence Dashboard

Dashboard interactivo para monitorear riesgo de default en préstamos LendingClub (2007-2018).

## Requisitos

- Python 3.10+
- Pip

## Instalacion

```bash
pip install -r requirements.txt
```

### API key FRED (datos macroeconomicos)

El proyecto usa la API de la Reserva Federal (FRED) para enriquecer el dataset con:
- Tasa de desempleo (`UNRATE`)
- Tasa de interes de la Fed (`FEDFUNDS`)
- Inflacion CPI (`CPIAUCSL`)

La API key ya esta configurada en `scripts/fetch_fred.py`. Si deseas usar la tuya, registrate en https://fred.stlouisfed.org/docs/api/api_key.html

## Datos

### Dataset completo (raw)

Los archivos originales deben ubicarse en `data/raw/`:

| Archivo | Tamano | Filas | Contenido |
|---------|--------|-------|-----------|
| `accepted_2007_to_2018Q4.csv` | ~1.6 GB | ~2.26M | Prestamos otorgados (151 columnas) |
| `rejected_2007_to_2018Q4.csv` | ~1.8 GB | ~27.7M | Solicitudes rechazadas (9 columnas) |

### Muestra (sample)

Si no se dispone del dataset completo, usar muestras de 5.000 filas en `data/interim/`:

```
data/interim/
  accepted_sample_5k.csv
  rejected_sample_5k.csv
```

Estas muestras se generan automaticamente si no existen al ejecutar el preprocessing.

### Generar muestras manualmente

```bash
python scripts/generate_samples.py -n 20000 --seed 42
```

Argumentos:
- `-n / --num_rows`: numero de filas a muestrear (default: 5000)
- `--seed`: semilla aleatoria (default: 42)
- `--gz`: leer archivos comprimidos `.gz`
- Las muestras se guardan en `data/interim/` con nombre `{tipo}_sample_{n}.csv`

## Pipeline completo

### 0. Datos macroeconomicos (FRED API)

El pipeline integra indicadores macroeconomicos via API de la Reserva Federal (FRED).

```bash
python scripts/fetch_fred.py
```

Descarga:
- `UNRATE` — Tasa de desempleo
- `FEDFUNDS` — Tasa de interes de la Fed
- `CPIAUCSL` — Inflacion (CPI)

Guarda `data/external/fred_indicators.csv` para ser mergeado por fecha en el ETL.

### 1. Preprocesamiento

Procesa el archivo `accepted_sample_{n}.csv` (sample) o `accepted_2007_to_2018Q4.csv` (full) y genera los CSVs limpios en `data/processed/`. Durante el ETL, mergea automaticamente los indicadores FRED por (year, month) de `issue_d`.

```bash
python src/preprocessing.py
```

Configuracion en `src/preprocessing.py`:
- `SAMPLE_SIZE = 20000` — tamano del sample (default)
- `sample=True` en `run_pipeline()` — usar sample; cambiar a `False` para datos completos

Output:
- `data/processed/accepted_clean.csv` (~20.000 filas, 118 columnas: 115 originales + unrate, fed_funds, cpi)
- `data/processed/rejected_clean.csv`
- `data/processed/combined_summary.csv`

### 2. Entrenar modelo

```bash
python scripts/train_model.py
```

- Lee `data/processed/accepted_clean.csv` (incluye variables macroeconomicas)
- Entrena pipeline XGBoost con preprocesamiento (imputacion + scaling + one-hot)
- Guarda modelo en `models/xgboost_pipeline.pkl`
- Guarda metricas en `models/model_metadata.json`

**Comparativa de metricas:**

| Metrica | Sin FRED | Con FRED | Mejora |
|---------|----------|----------|--------|
| AUC-ROC | 0.7320 | 0.7558 | +0.0238 |
| Recall | 0.6501 | 0.6984 | +0.0483 |
| F1 | 0.3529 | 0.3657 | +0.0128 |

### 3. Ejecutar dashboard

```bash
streamlit run dashboard/app.py
```

Navegar a http://localhost:8501

## Estructura del proyecto

```
archive/
  data/
    raw/                  # CSV originales (no modificar)
    interim/              # Muestras generadas
    processed/            # Datos limpios listos para modelado
    external/             # Datos de fuentes externas (FRED API)
  dashboard/
    app.py                # Entry point del dashboard
    views/                # Paginas del dashboard (7 secciones)
    utils/                # Data loader, charts, model loader
  scripts/
    generate_samples.py   # Generador de muestras desde raw
    train_model.py        # Entrenamiento XGBoost
  src/
    preprocessing.py      # Pipeline ETL
  models/
    xgboost_pipeline.pkl  # Modelo entrenado
    model_metadata.json   # Metricas del modelo
  notebooks/              # Jupyter notebooks (EDA, modelado)
  requirements.txt
  README.md
```

## Dashboard: 7 secciones

| # | Seccion | Descripcion |
|---|---------|-------------|
| 01 | Riesgo y Calidad de Cartera | KPIs globales, default por grade, FICO, DTI, serie temporal |
| 02 | Rentabilidad / Pricing | Heatmap term x grade, tasa vs default, tabla resumen |
| 03 | Perfil del Solicitante | Default por proposito, ingresos, vivienda, estado |
| 04 | Aceptados vs Rechazados | Comparacion de metricas entre grupos |
| 05 | Dificultad Financiera | Hardship, settlement, morosidad |
| 06 | Salud del Pipeline | Metricas tecnicas, volumen, datos faltantes |
| 07 | Prediccion de Default | Formulario interactivo con scoring XGBoost |

## Modo de uso: Datos completos vs Muestra

### Con muestra (rapido, ~5-20k filas)
- Ideal para desarrollo y pruebas
- Los archivos sample ya estan en `data/interim/`
- `preprocessing.py` configurado con `sample=True` por defecto

### Con datos completos (~2.26M filas)
- Requiere ~3-4 GB de RAM y varios minutos de procesamiento
- Cambiar `sample=True` a `sample=False` en `src/preprocessing.py:run_pipeline()`
- `scripts/generate_samples.py` permite crear muestras de cualquier tamano para pruebas controladas
