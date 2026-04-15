# ⚡ Ecuador Energy Anomalies

> Detección multi-técnica de anomalías en el sector eléctrico latinoamericano. Isolation Forest + STL + CUSUM, validado contra la crisis energética de Ecuador 2024. 8 países, 784 meses de datos reales.

[![English](https://img.shields.io/badge/language-English-blue)](../README.md) [![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

---

## Resultados Clave

![Demanda con anomalías](../docs/images/01_ecuador_demanda_anomalias.png)

**Ground Truth Estricto (oct-dic 2024, 3 meses — crisis severa)**

| Métrica | IF | STL | CUSUM | **Consenso** | Ponderado |
|---------|-----|-----|-------|-------------|-----------|
| Precision | 50.0% | 28.6% | 18.8% | **60.0%** | 60.0% |
| Recall | 100% | 66.7% | 100% | **100%** | 100% |
| F1-Score | 0.667 | 0.400 | 0.316 | **0.750** | 0.750 |
| MCC | 0.694 | 0.407 | 0.397 | **0.765** | 0.765 |

**Ground Truth Amplio (abr-dic 2024, Decreto Ejecutivo 229)**

| Métrica | IF | STL | CUSUM | **Consenso** | Ponderado |
|---------|-----|-----|-------|-------------|-----------|
| Precision | 50.0% | 42.9% | 18.8% | **60.0%** | 60.0% |
| Recall | 33.3% | 33.3% | 33.3% | 33.3% | 33.3% |
| F1-Score | 0.400 | 0.375 | 0.240 | **0.429** | 0.429 |

> Métricas de `metrics.json`. Reproducir: `python scripts/train_model.py`

---

## Comparación: 9 Modelos

![9 baselines](../docs/images/19_baselines_9modelos.png)

| Modelo | Precision | Recall | F1 | MCC |
|--------|-----------|--------|-----|-----|
| **Consenso ≥2** | **0.600** | **1.000** | **0.750** | **0.763** |
| IF por país | 0.500 | 1.000 | 0.667 | 0.692 |
| LOF | 0.500 | 1.000 | 0.667 | 0.692 |
| LSTM-AE | 0.273 | 1.000 | 0.429 | 0.491 |
| Prophet | 0.333 | 0.667 | 0.444 | 0.441 |
| Elliptic Envelope | 0.167 | 0.333 | 0.222 | 0.189 |
| DBSCAN | 0.056 | 1.000 | 0.105 | 0.123 |
| One-Class SVM | 0.056 | 0.333 | 0.095 | 0.042 |
| ARIMA | 0.000 | 0.000 | 0.000 | 0.000 |

---

## Hallazgo Clave: Dependencia Hidro Determina Efectividad

![Hydro vs F1](../docs/images/29_hydro_vs_f1.png)

| País | Hidro % | F1 Consenso | Tipo Crisis |
|------|---------|-------------|-------------|
| Brazil | 47.9% | **0.727** | Sequía (hídrica) |
| Ecuador | 38.1% | **0.750** | Sequía (hídrica) |
| Colombia | 41.1% | 0.429 | El Niño (hídrica) |
| Chile | 14.3% | 0.000 | Mega-sequía (no eléctrica) |
| Argentina | 13.6% | 0.000 | Ola de calor (térmica) |

---

## Validación Estadística

| Variable | μ Normal | μ Anomalía | p-value | Cohen's d | Efecto |
|----------|----------|------------|---------|-----------|--------|
| gen_hydro | 38.71% | 28.72% | 0.0004*** | 2.81 | MUY GRANDE |
| gen_fossil | 11.87% | 21.47% | 0.0004*** | 3.14 | MUY GRANDE |
| co2_intensity | 175.7 | 298.5 | 0.0003*** | 3.15 | MUY GRANDE |
| importaciones | 0.04 | 0.16 | 0.0045** | 1.37 | GRANDE |
| demanda_twh | 2.86 | 2.87 | 0.9702 ns | 0.04 | NULO |

McNemar Consenso vs IF: p=1.000 (no significativo con n=73).

---

## Limitaciones (Honestas)

- McNemar p=1.0: No se puede probar estadísticamente consenso > IF con n=73
- Bootstrap CI amplio: F1=[0.400, 1.000] por N pequeño
- Falla en países con baja hidro (Chile F1=0, Argentina F1=0)
- GT amplio recall=33%: Solo detecta los 3 meses pico de 9
- 85 meses Ecuador es el mínimo viable
- Datos mensuales suavizan crisis graduales

## Fuentes Oficiales (Ground Truth)

| País | Crisis | Fuente Oficial |
|------|--------|----------------|
| Ecuador | Abr-Dic 2024 | Decreto Ejecutivo 229; Informe Anual CENACE 2024 |
| Brasil | Jun-Nov 2021 | Decreto 10.939/2021; MP 1.055/2021 |
| Colombia | Ene-Jun 2024 | Informes XM Colombia; precios +22.68% |
| Chile | 2019 (agudo) | U. de Chile; Biblioteca del Congreso Nacional |
| Argentina | Ene-Mar 2022 | SMN Argentina; FARN informe climático |

---

**Diego Fernando Lojan Tenesaca** — Data & AI Engineer
[![GitHub](https://img.shields.io/badge/GitHub-DiegoFernandoLojanTenesaca-181717?logo=github)](https://github.com/DiegoFernandoLojanTenesaca) [![LinkedIn](https://img.shields.io/badge/LinkedIn-diego--fernando--lojan-0A66C2?logo=linkedin)](https://linkedin.com/in/diego-fernando-lojan)
