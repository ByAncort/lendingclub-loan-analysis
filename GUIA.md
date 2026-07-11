
# generacion de csv limpios

python src/preprocessing.py

Pipeline del script:
1. Carga data/interim/accepted_sample_5k.csv (sample) o data/raw/accepted_2007_to_2018Q4.csv (full)
2. Limpieza: normaliza nombres de columnas a snake_case
3. Mapea loan_status a target binario bad_loan (1=Default/Charged Off/Late, 0=Fully Paid/Current)
4. Parsea fechas (issue_d, earliest_cr_line), montos (string $ -> float), porcentajes (int_rate, revol_util), plazos (term -> meses)
5. Feature engineering: fico_score (midpoint), had_hardship, debt_settlement
6. Elimina 42 columnas de leakage (datos posteriores al préstamo: out_prncp, total_pymnt, recoveries, last_pymnt, etc.)
7. Output: data/processed/accepted_clean.csv (5,000 filas, 113 columnas)
Para regenerar con datos completos (no sample), cambia sample=True a sample=False en run_pipeline().