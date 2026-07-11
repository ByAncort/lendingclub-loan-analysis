# Estructura del Proyecto — LendingClub Loans (2007-2018)

```
archive/
│
├── data/
│   ├── raw/                              # Datos originales (no modificar)
│   │   ├── accepted_2007_to_2018Q4.csv       # ~1.6 GB — 151 columnas
│   │   ├── accepted_2007_to_2018Q4.csv.gz    # ~392 MB comprimido
│   │   ├── rejected_2007_to_2018Q4.csv       # ~1.8 GB — 9 columnas
│   │   └── rejected_2007_to_2018Q4.csv.gz    # ~255 MB comprimido
│   │
│   ├── interim/                          # Datos intermedios / muestras
│   │   ├── accepted_sample_5k.csv            # Muestra 5k filas para exploración rápida
│   │   └── rejected_sample_5k.csv            # Muestra 5k filas para exploración rápida
│   │
│   └── processed/                        # Datos limpios listos para modelado
│
├── notebooks/                            # Jupyter notebooks (EDA, feature engineering, modelos)
│
├── src/                                  # Código fuente / scripts
│
├── reports/
│   └── figures/                          # Gráficos y visualizaciones
│
├── models/                               # Modelos entrenados guardados
│
└── references/                           # Data dictionaries, documentación externa
```

## Resumen de Datos

| Dataset | Filas | Columnas | Contenido |
|---------|-------|----------|-----------|
| `accepted` | ~2.26M | 151 | Préstamos otorgados con outcome (loan_status), perfil crediticio, tasas, etc. |
| `rejected` | ~27.7M | 9 | Solicitudes rechazadas (solo monto, fecha, riesgo, DTI, ubicación) |

## Plan de Desarrollo — Evaluación Final Transversal

### Información General

| Ítem | Detalle |
|------|---------|
| **Asignatura** | Programación para la Ciencia de Datos |
| **Fecha** | Miércoles 15 de julio de 2026 — 19:01 a 22:00 hrs |
| **Sala** | V108 |
| **Ponderación** | 40% |
| **Tipo** | Proyecto integrador — simulación entorno profesional |

### Objetivo del Proyecto

Diseñar, implementar y presentar una solución completa de ciencia de datos que integre:

- Gestión avanzada de datos
- Procesos ETL
- Modelos de machine learning
- Visualizaciones interactivas
- Despliegue profesional

### Evaluación: 2 Partes

| Parte | Descripción |
|-------|-------------|
| **Encargo (entrega técnica)** | Proyecto completo: estructura, pipeline ETL, modelos ML, dashboards, documentación y evidencias |
| **Presentación (15 min)** | Problema, arquitectura, demo funcional, métricas, visualizaciones + preguntas individuales |

> **Nota individual** — cada estudiante recibe nota según su desempeño, dominio del proyecto, claridad en presentación y capacidad de explicar su aporte personal.

### Entregable

Enviar mensaje en AVA al profesor con:

- Proyecto desarrollado (código, documentación, evidencias)
- Archivos necesarios para revisar la solución

### Dataset

> El dataset debe ser **seleccionado por el equipo**, **nuevo y distinto** a los usados en el curso. Debe permitir: caso realista, integración de fuentes, ETL, ML y dashboards.

### Cronograma del Proyecto

| Fase | Actividades | Entregable |
|------|-------------|------------|
| **S1** (27-30 Jun) | Exploración del dataset, análisis de variables, definición del problema de negocio | `notebooks/01_eda.ipynb` + documento de alcance en `references/` |
| **S2** (1-4 Jul) | Pipeline ETL: limpieza, transformaciones, feature engineering, integrar fuentes externas (API) | `src/etl.py` + `data/processed/` |
| **S3** (5-8 Jul) | Modelos ML: entrenamiento, validación, comparación de algoritmos, selección del modelo final | `notebooks/02_modeling.ipynb` + `models/` |
| **S4** (9-12 Jul) | Dashboard interactivo, visualizaciones, preparación de presentación | `src/dashboard.py` + `reports/figures/` |
| **S5** (13-15 Jul) | Documentación final, pruebas, empaquetado (Docker), ensayo presentación | `ESTRUCTURA.md` actualizado, `Dockerfile`, `README.md` |

### Stack Tecnológico Sugerido

| Herramienta | Uso |
|-------------|-----|
| Python + Pandas | Procesamiento y ETL |
| Scikit-learn | Modelos ML |
| FastAPI / Flask | API de despliegue |
| Streamlit / Dash | Dashboard interactivo |
| Git + GitHub | Control de versiones y colaboración |
| Docker | Contenedorización y despliegue |
| GitHub Actions / CI/CD | Integración continua |

### Rúbrica de Evaluación (Encargo)

| Criterio | Ponderación |
|----------|-------------|
| Organización del proyecto y estructura de carpetas | 15% |
| Calidad del pipeline ETL (limpieza, transformaciones, integración fuentes) | 25% |
| Modelos de machine learning (selección, métricas, validación) | 25% |
| Dashboards y visualizaciones interactivas | 20% |
| Documentación, evidencias y reproducibility | 15% |

### Checklist de Preparación

- [ ] Dataset seleccionado y aprobado (nuevo, distinto al curso)
- [ ] EDA completo con visualizaciones exploratorias
- [ ] Pipeline ETL funcional y documentado
- [ ] Al menos 2 modelos ML entrenados y comparados
- [ ] Dashboard interactivo desplegable
- [ ] Repositorio Git con commits frecuentes
- [ ] Dockerfile para contenedorización
- [ ] Documentación (ESTRUCTURA.md, README.md)
- [ ] Presentación preparada (15 min + demo)
- [ ] Cada integrante preparado para preguntas individuales
