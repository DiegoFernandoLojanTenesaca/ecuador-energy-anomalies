# ⚡ Ecuador Energy Anomalies

> Detección de anomalías multi-técnica en el sector eléctrico de Latinoamérica. Isolation Forest + STL Decomposition + CUSUM, validado contra la crisis energética de Ecuador 2024. 8 países, 784 meses de datos reales.

[![English](https://img.shields.io/badge/language-English-blue)](../README.md) [![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

---

## ¿Por qué este proyecto?

Ecuador genera ~70% de su electricidad con **hidroeléctricas**. Las sequías causan apagones de hasta 14 horas. Este proyecto detecta crisis energéticas automáticamente usando 3 técnicas complementarias y explica **por qué** cada anomalía fue marcada.

### Resultados Clave

![Demanda con anomalías](images/01_ecuador_demanda_anomalias.png)

| Métrica | IF por país | STL | CUSUM | **Consenso ≥2** |
|---------|------------|-----|-------|----------------|
| Precision | 50.0% | 28.6% | 18.8% | **60.0%** |
| Recall | 100% | 66.7% | 100% | **100%** |
| F1-Score | 66.7% | 40.0% | 31.6% | **75.0%** |
| MCC | 0.694 | 0.407 | 0.397 | **0.765** |

> Métricas reproducibles desde `data/processed/metrics.json` via `python scripts/train_model.py`

---

## Datos: 8 Países, 784 Meses Reales

![Dependencia hidro por país](images/06_hidro_por_pais.png)

---

## Enfoque de 3 Técnicas

![Comparación multi-técnica](images/03_multitecnica_comparacion.png)

| Técnica | Qué detecta | Crisis Ecuador oct-dic 2024 |
|---------|-------------|---------------------------|
| **Isolation Forest** | Outliers multivariados | 3/3 detectados |
| **STL Decomposition** | Residuales > 2σ tras quitar tendencia | 2/3 detectados |
| **CUSUM** | Cambios estructurales en generación hidro | 3/3 detectados |
| **Consenso ≥2** | Confirmado por al menos 2 métodos | **3/3 detectados** |

![Métricas comparativas](images/05_metricas_comparativas.png)

---

## Mix Energético de Ecuador

![Mix energético](images/02_ecuador_mix_energetico.png)

![Intensidad CO2](images/08_co2_intensity.png)

---

## Análisis Multi-País

![Heatmap anomalías por país](images/04_heatmap_paises.png)

![LATAM comparativo hidro](images/09_latam_hidro_comparativo.png)

---

## Matriz de Confusión

![Matriz confusión](images/07_confusion_matrix.png)

---

## Validación Estadística

| Variable | p-value | Cohen's d | Efecto |
|----------|---------|-----------|--------|
| gen_hydro | 0.002** | 1.29 | GRANDE |
| gen_fossil | 0.003** | 1.30 | GRANDE |
| co2_intensity | 0.008** | 1.22 | GRANDE |

Estabilidad: CV = 0.0% en 20 semillas aleatorias.

---

## Notebooks de Análisis

| # | Notebook | Descripción |
|---|----------|-------------|
| 00 | [Origen y Diccionario](../notebooks/EDA/00_origen_y_diccionario_datos.ipynb) | Fuentes, 20 variables documentadas |
| 01 | [Carga y Exploración](../notebooks/EDA/01_carga_y_exploracion.ipynb) | Estructura, primeras visualizaciones |
| 02 | [Limpieza y Calidad](../notebooks/EDA/02_limpieza_y_calidad.ipynb) | Nulos, outliers, consistencia |
| 03 | [Análisis de Patrones](../notebooks/EDA/03_analisis_patrones.ipynb) | Tendencias, estacionalidad, correlaciones |
| 04 | [Feature Engineering](../notebooks/EDA/04_feature_engineering.ipynb) | 24 → 213 features |
| 05 | [Selección de Modelo](../notebooks/EDA/05_seleccion_modelo.ipynb) | IF vs LOF vs SVM |
| 06 | [Entrenamiento](../notebooks/EDA/06_entrenamiento_evaluacion.ipynb) | Modelo final, SHAP |
| 07 | [Tuning](../notebooks/EDA/07_tuning_hiperparametros.ipynb) | Optuna, TimeSeriesSplit |
| 08 | [Validación](../notebooks/EDA/08_validacion_metricas.ipynb) | Tests estadísticos, bootstrap |

---

## Limitaciones (Honestas)

- Datos mensuales suavizan crisis graduales (sequía 2023: solo 2/4 detectada por STL)
- Apagones programados (abr-jun 2024) no cambian volúmenes → indetectables
- 85 meses reales para Ecuador es el mínimo viable
- Silhouette=0.23 refleja el tamaño de muestra, no falla del modelo

## Referencias

- Liu et al. (2008). *Isolation Forest*. IEEE ICDM.
- Cleveland et al. (1990). *STL: A Seasonal-Trend Decomposition*.
- Page (1954). *Continuous Inspection Schemes*. Biometrika (CUSUM).
- Chandola et al. (2009). *Anomaly Detection: A Survey*. ACM.

---

**Diego Fernando Lojan Tenesaca** — Data & AI Engineer
[![GitHub](https://img.shields.io/badge/GitHub-DiegoFernandoLojanTenesaca-181717?logo=github)](https://github.com/DiegoFernandoLojanTenesaca) [![LinkedIn](https://img.shields.io/badge/LinkedIn-diego--fernando--lojan-0A66C2?logo=linkedin)](https://linkedin.com/in/diego-fernando-lojan)