"""Genera PDFs profesionales detallados del estudio (ES + EN)."""

import json
import os
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "images"
DATA = ROOT / "data" / "processed"

with open(DATA / "metrics.json") as f:
    M = json.load(f)
with open(DATA / "baselines_comparison.json") as f:
    BL = json.load(f)
with open(DATA / "confidence_intervals.json") as f:
    CI = json.load(f)
with open(DATA / "mcnemar_tests.json") as f:
    MC = json.load(f)

MS = M["strict"]
MB = M["broad_decree"]


def clean(t):
    for a, b in [
        ("\u2014", "-"), ("\u2013", "-"), ("\u00f1", "n"), ("\u00e9", "e"),
        ("\u00e1", "a"), ("\u00ed", "i"), ("\u00f3", "o"), ("\u00fa", "u"),
        ("\u00fc", "u"), ("\u03c3", "sigma"), ("\u2265", ">="), ("\u2264", "<="),
        ("\u00b2", "2"),
    ]:
        t = t.replace(a, b)
    return t


class Report(FPDF):
    def __init__(self, lang="en"):
        super().__init__()
        self.lang = lang
        self.ES = lang == "es"

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        h = "Ecuador Energy Anomalies - Reporte Tecnico" if self.ES else "Ecuador Energy Anomalies - Technical Report"
        self.cell(0, 6, clean(h), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"{self.page_no()}/{{nb}}", align="C")

    def cover(self):
        self.add_page()
        self.ln(50)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(21, 101, 192)
        self.cell(0, 16, "Ecuador Energy Anomalies", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        self.set_font("Helvetica", "", 15)
        self.set_text_color(80, 80, 80)
        sub = "Deteccion Multi-Tecnica de Anomalias en el" if self.ES else "Multi-Technique Anomaly Detection in"
        sub2 = "Sector Electrico de Latinoamerica" if self.ES else "Latin America's Electricity Sector"
        self.cell(0, 8, clean(sub), align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, clean(sub2), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)
        self.set_draw_color(21, 101, 192)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(12)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, "Diego Fernando Lojan Tenesaca", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 11)
        self.cell(0, 7, "Data & AI Engineer", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "github.com/DiegoFernandoLojanTenesaca", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 6, "linkedin.com/in/diego-fernando-lojan", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(25)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        kw = "8 paises | 784 meses | 9 modelos | 3 tecnicas consenso" if self.ES else "8 countries | 784 months | 9 models | 3 consensus techniques"
        self.cell(0, 5, clean(kw), align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, f"F1 = {MS['Consensus']['f1']:.3f}  |  MCC = {MS['Consensus']['mcc']:.3f}  |  Recall = {MS['Consensus']['recall']:.0%}", align="C", new_x="LMARGIN", new_y="NEXT")
        src = "Fuentes oficiales: Decreto 229, CENACE, XM Colombia, EPE Brasil" if self.ES else "Official sources: Decree 229, CENACE, XM Colombia, EPE Brazil"
        self.cell(0, 5, clean(src), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(15)
        fecha = "Abril 2026" if self.ES else "April 2026"
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, fecha, align="C", new_x="LMARGIN", new_y="NEXT")

    def section(self, num, title):
        self.add_page()
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(21, 101, 192)
        self.cell(0, 12, clean(f"{num}. {title}"), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(21, 101, 192)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(6)

    def sub(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, clean(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, clean(text))
        self.ln(3)

    def note(self, text):
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 5, clean(text))
        self.ln(2)

    def img(self, name, caption="", w=180):
        path = str(IMG / name)
        if not os.path.exists(path):
            return
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, clean(caption), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def t(self, key_es, key_en):
        return key_es if self.ES else key_en


def build_report(lang):
    r = Report(lang)
    r.alias_nb_pages()
    r.set_auto_page_break(auto=True, margin=18)
    NL = "\n"

    # === PORTADA ===
    r.cover()

    # === 1. RESUMEN ===
    r.section(1, r.t("Resumen Ejecutivo", "Executive Summary"))
    r.body(r.t(
        "Este estudio presenta un enfoque multi-tecnica para la deteccion automatica de anomalias "
        "en el sector electrico de Latinoamerica, con enfoque principal en la crisis energetica de "
        "Ecuador 2024. Se analizan 784 meses de datos reales de 8 paises, obtenidos de Ember Global "
        "Electricity Data (CC BY 4.0), sin datos sinteticos ni interpolados.",
        "This study presents a multi-technique approach for automatic anomaly detection in Latin "
        "America's electricity sector, focusing on Ecuador's 2024 energy crisis. We analyze 784 months "
        "of real data from 8 countries, sourced from Ember Global Electricity Data (CC BY 4.0), with "
        "no synthetic or interpolated data."
    ))
    r.body(r.t(
        "Se implementan tres tecnicas complementarias: Isolation Forest (deteccion multivariada), "
        "STL Decomposition (patron estacional) y CUSUM (cambio estructural). Un enfoque de consenso "
        "que requiere acuerdo de al menos 2 de 3 tecnicas reduce falsos positivos manteniendo alto recall.",
        "Three complementary techniques are implemented: Isolation Forest (multivariate detection), "
        "STL Decomposition (seasonal pattern), and CUSUM (structural change). A consensus approach "
        "requiring agreement from at least 2 of 3 techniques reduces false positives while maintaining "
        "high recall."
    ))
    r.sub(r.t("Resultados principales", "Key results"))
    mcon = MS["Consensus"]
    r.body(f"Consensus (GT strict oct-dec 2024): P={mcon['precision']:.1%}  R={mcon['recall']:.0%}  "
           f"F1={mcon['f1']:.3f}  MCC={mcon['mcc']:.3f}  |  Crisis: {mcon['crisis_detected']}/{mcon['crisis_total']}")
    mconb = MB["Consensus"]
    r.body(f"Consensus (GT broad Decree 229): P={mconb['precision']:.1%}  R={mconb['recall']:.1%}  "
           f"F1={mconb['f1']:.3f}  |  Crisis: {mconb['crisis_detected']}/{mconb['crisis_total']}")
    r.body(r.t(
        "El consenso supera a 8 baselines incluyendo LOF, SVM, DBSCAN, ARIMA, Prophet y LSTM-AE. "
        "Los tests estadisticos confirman diferencias significativas (gen_hydro p=0.0004***, Cohen d=2.81).",
        "The consensus outperforms 8 baselines including LOF, SVM, DBSCAN, ARIMA, Prophet, and LSTM-AE. "
        "Statistical tests confirm significant differences (gen_hydro p=0.0004***, Cohen's d=2.81)."
    ))
    r.img("17_pipeline_infografia.png", r.t("Figura 1: Pipeline del estudio", "Figure 1: Study pipeline"))

    # === 2. CONTEXTO ===
    r.section(2, r.t("Contexto y Problema", "Context and Problem"))
    r.body(r.t(
        "Ecuador genera aproximadamente el 70% de su electricidad a partir de fuentes hidroelectricas, "
        "principalmente las centrales Coca Codo Sinclair (1,500 MW) y Paute (1,075 MW). Esta alta "
        "dependencia del recurso hidrico crea una vulnerabilidad critica ante sequias y variabilidad "
        "climatica, particularmente ante el fenomeno de El Nino.",
        "Ecuador generates approximately 70% of its electricity from hydroelectric sources, primarily "
        "the Coca Codo Sinclair (1,500 MW) and Paute (1,075 MW) plants. This heavy dependence on water "
        "resources creates a critical vulnerability to droughts and climate variability, particularly "
        "El Nino events."
    ))
    r.sub(r.t("Cronologia de la crisis 2024", "2024 crisis timeline"))
    r.body(r.t(
        "- Abril 2024: Decreto Ejecutivo No. 229 declara estado de excepcion en el sector electrico\n"
        "- Abril-Junio 2024: Apagones programados de hasta 8 horas diarias\n"
        "- Octubre-Diciembre 2024: Crisis maxima con racionamientos de hasta 14 horas\n"
        "- Impacto: peor sequia en 61 anios, 1.5% del PIB (USD 1,700 millones)\n"
        "- Diciembre 2025: Recaida con aumento de generacion fosil",
        "- April 2024: Executive Decree No. 229 declares state of exception in the electricity sector\n"
        "- April-June 2024: Programmed blackouts of up to 8 hours daily\n"
        "- October-December 2024: Peak crisis with rolling blackouts up to 14 hours\n"
        "- Impact: worst drought in 61 years, 1.5% of GDP (USD 1.7 billion)\n"
        "- December 2025: Recurrence with increased fossil generation"
    ))
    r.img("02_ecuador_mix_energetico.png", r.t("Figura 2: Mix energetico de Ecuador - hidro cae, fosil compensa durante crisis", "Figure 2: Ecuador energy mix - hydro drops, fossil compensates during crisis"))
    r.sub(r.t("Contexto regional", "Regional context"))
    r.body(r.t(
        "La vulnerabilidad hidroelectrica no es exclusiva de Ecuador. Varios paises latinoamericanos "
        "enfrentan riesgos similares: Brasil (48% hidro, crisis 2021), Colombia (41% hidro, impacto "
        "El Nino 2024), Chile (mega-sequia desde 2010). Esto justifica un estudio comparativo regional "
        "en lugar de un analisis aislado de un solo pais.",
        "Hydroelectric vulnerability is not exclusive to Ecuador. Several Latin American countries face "
        "similar risks: Brazil (48% hydro, 2021 crisis), Colombia (41% hydro, El Nino impact 2024), "
        "Chile (mega-drought since 2010). This justifies a regional comparative study rather than an "
        "isolated single-country analysis."
    ))
    r.img("06_hidro_por_pais.png", r.t("Figura 3: Dependencia hidroelectrica por pais", "Figure 3: Hydroelectric dependency by country"))

    # === 3. DATOS ===
    r.section(3, r.t("Fuentes de Datos", "Data Sources"))
    r.body(r.t(
        "Todos los datos provienen de fuentes internacionales verificadas y de acceso publico. "
        "No se utilizaron datos sinteticos, interpolados ni simulados.",
        "All data comes from verified, publicly available international sources. No synthetic, "
        "interpolated, or simulated data was used."
    ))
    r.sub("Ember Global Electricity Data")
    r.body(r.t(
        "Fuente principal. Datos mensuales de generacion por fuente, demanda, emisiones CO2 e "
        "importaciones netas. Licencia CC BY 4.0. Cobertura: 8 paises latinoamericanos, 784 meses "
        "en total. Los datos de Ember son recopilados de fuentes nacionales oficiales (CENACE para "
        "Ecuador, XM para Colombia, ONS para Brasil) y validados contra IEA y Naciones Unidas.",
        "Primary source. Monthly data on generation by source, demand, CO2 emissions, and net imports. "
        "CC BY 4.0 license. Coverage: 8 Latin American countries, 784 total months. Ember data is "
        "collected from official national sources (CENACE for Ecuador, XM for Colombia, ONS for Brazil) "
        "and cross-validated against IEA and United Nations."
    ))
    r.sub(r.t("Cobertura por pais", "Coverage by country"))
    r.body("Ecuador: 85m | Colombia: 120m | Peru: 86m | Chile: 123m\n"
           "Brazil: 99m | Argentina: 98m | Bolivia: 86m | Uruguay: 87m")
    r.img("11_correlacion_paises.png", r.t("Figura 4: Correlacion de generacion hidro entre paises", "Figure 4: Hydro generation correlation between countries"))

    # === 4. METODOLOGIA ===
    r.section(4, r.t("Metodologia", "Methodology"))
    r.body(r.t(
        "En lugar de depender de un unico modelo, se implementan tres tecnicas complementarias "
        "que capturan diferentes tipos de anomalias. Se requiere consenso (acuerdo de al menos 2 "
        "de 3 tecnicas) para clasificar un mes como anomalo.",
        "Instead of relying on a single model, three complementary techniques are implemented that "
        "capture different types of anomalies. Consensus (agreement from at least 2 of 3 techniques) "
        "is required to classify a month as anomalous."
    ))
    r.sub("4.1 Isolation Forest (Liu et al., 2008)")
    r.body(r.t(
        "Algoritmo no supervisado basado en arboles que aisla anomalias mediante particiones aleatorias. "
        "La intuicion: los puntos anomalos necesitan MENOS particiones para quedar aislados. Se aplica "
        "POR PAIS (no globalmente) para evitar que paises grandes como Brasil dominen la deteccion. "
        "Parametros: n_estimators=300, contamination=0.08. Features: 55 variables por pais (rolling "
        "statistics 6/12 meses, lags, z-scores, ratios, temporales).",
        "Unsupervised tree-based algorithm that isolates anomalies through random partitioning. The "
        "intuition: anomalous points require FEWER partitions to become isolated. Applied PER COUNTRY "
        "(not globally) to prevent large countries like Brazil from dominating detection. Parameters: "
        "n_estimators=300, contamination=0.08. Features: 55 variables per country (rolling statistics "
        "6/12 months, lags, z-scores, ratios, temporal)."
    ))
    r.sub("4.2 STL Decomposition (Cleveland et al., 1990)")
    r.body(r.t(
        "La descomposicion Seasonal-Trend using LOESS separa la serie de generacion hidro en tres "
        "componentes: tendencia (direccion a largo plazo), estacionalidad (patron anual seca/lluviosa) "
        "y residual (variacion no explicada). Meses con residuales que exceden 2 desviaciones estandar "
        "se marcan como anomalias. Particularmente efectiva para detectar crisis graduales que IF no capta.",
        "Seasonal-Trend decomposition using LOESS separates the hydro generation series into three "
        "components: trend (long-term direction), seasonality (annual wet/dry pattern), and residual "
        "(unexplained variation). Months with residuals exceeding 2 standard deviations are flagged "
        "as anomalies. Particularly effective at detecting gradual crises that IF misses."
    ))
    r.sub("4.3 CUSUM (Page, 1954)")
    r.body(r.t(
        "El grafico de control de Suma Acumulada detecta cambios sostenidos en la media de la generacion "
        "hidro. Parametros: tolerancia k=0.5*sigma, umbral de alarma h=4*sigma. Efectivo para detectar "
        "cambios de regimen prolongados, no solo picos puntuales.",
        "The Cumulative Sum control chart detects sustained shifts in the mean of hydro generation. "
        "Parameters: allowance k=0.5*sigma, alarm threshold h=4*sigma. Effective at detecting prolonged "
        "regime changes, not just point spikes."
    ))
    r.sub(r.t("4.4 Consenso ponderado por dependencia hidro (Contribucion novel)",
              "4.4 Hydro-weighted consensus (Novel contribution)"))
    r.body(r.t(
        "Contribucion original de este estudio. Los pesos de cada tecnica se adaptan segun la dependencia "
        "hidroelectrica del pais:\n\n"
        "  w_stl = w_cusum = 0.5 + 0.5 * alpha\n"
        "  w_if = 1.0 - 0.3 * alpha\n"
        "  alpha = min(hydro_share * 2, 1.0)\n\n"
        "Paises con alta hidro (Ecuador 38%, Brasil 48%) dan mas peso a STL/CUSUM (especializados en "
        "patrones hidro). Paises con baja hidro (Chile 14%) dan mas peso a IF (multivariado, no depende "
        "de hidro). Esto produce un consenso adaptado al perfil energetico de cada pais.",
        "Original contribution of this study. Each technique's weight adapts based on the country's "
        "hydroelectric dependency:\n\n"
        "  w_stl = w_cusum = 0.5 + 0.5 * alpha\n"
        "  w_if = 1.0 - 0.3 * alpha\n"
        "  alpha = min(hydro_share * 2, 1.0)\n\n"
        "High-hydro countries (Ecuador 38%, Brazil 48%) give more weight to STL/CUSUM (specialized in "
        "hydro patterns). Low-hydro countries (Chile 14%) give more weight to IF (multivariate, not "
        "hydro-dependent). This produces consensus adapted to each country's energy profile."
    ))
    r.img("03_multitecnica_comparacion.png", r.t("Figura 5: Tres tecnicas aplicadas a Ecuador", "Figure 5: Three techniques applied to Ecuador"))

    # === 5. BASELINES ===
    r.section(5, r.t("Comparacion: 9 Modelos", "Comparison: 9 Models"))
    r.body(r.t(
        "Se comparan 9 modelos de deteccion de anomalias sobre los datos de Ecuador con ground truth "
        "estricto (oct-dic 2024, 3 meses de crisis severa confirmada por Decreto Ejecutivo 229).",
        "9 anomaly detection models are compared on Ecuador data with strict ground truth (oct-dec 2024, "
        "3 months of severe crisis confirmed by Executive Decree 229)."
    ))
    r.img("19_baselines_9modelos.png", r.t("Figura 6: Comparacion de 9 modelos", "Figure 6: 9-model comparison"))
    r.img("24_tabla_9modelos.png", r.t("Figura 7: Tabla detallada de metricas", "Figure 7: Detailed metrics table"))
    r.img("25_radar_top5.png", r.t("Figura 8: Radar de los 5 mejores modelos", "Figure 8: Top 5 models radar"))
    r.note(r.t("Reproducible: python scripts/run_full_comparison.py", "Reproducible: python scripts/run_full_comparison.py"))

    # === 6. RESULTADOS ECUADOR ===
    r.section(6, r.t("Resultados: Ecuador", "Results: Ecuador"))
    r.img("01_ecuador_demanda_anomalias.png", r.t("Figura 9: Demanda con anomalias del consenso", "Figure 9: Demand with consensus anomalies"))
    r.sub(r.t("Metricas con dos ground truths", "Metrics with two ground truths"))
    r.body(r.t(
        "Se evaluan las metricas con dos definiciones de ground truth:\n\n"
        "GT ESTRICTO (oct-dic 2024, 3 meses): Solo los meses de crisis severa confirmada.\n"
        "  Consenso: P=60% R=100% F1=0.750 MCC=0.765 | Crisis: 3/3\n\n"
        "GT AMPLIO (abr-dic 2024, 9 meses): Todo el periodo del Decreto 229.\n"
        "  Consenso: P=60% R=33% F1=0.429 MCC=0.401 | Crisis: 3/9\n\n"
        "La diferencia en recall (100% vs 33%) refleja que el modelo detecta los 3 meses MAS SEVEROS "
        "pero no los 6 meses de crisis mas moderada (abril-septiembre). Esto es una limitacion de los "
        "datos mensuales: la crisis gradual se diluye.",
        "Metrics are evaluated with two ground truth definitions:\n\n"
        "STRICT GT (oct-dec 2024, 3 months): Only confirmed severe crisis months.\n"
        "  Consensus: P=60% R=100% F1=0.750 MCC=0.765 | Crisis: 3/3\n\n"
        "BROAD GT (apr-dec 2024, 9 months): Full Decree 229 period.\n"
        "  Consensus: P=60% R=33% F1=0.429 MCC=0.401 | Crisis: 3/9\n\n"
        "The recall difference (100% vs 33%) reflects that the model detects the 3 MOST SEVERE months "
        "but not the 6 months of more moderate crisis (April-September). This is a limitation of monthly "
        "data: gradual crises are smoothed out."
    ))
    r.img("18_tabla_metricas.png", r.t("Figura 10: Metricas por tecnica", "Figure 10: Metrics by technique"))
    r.img("07_confusion_matrix.png", r.t("Figura 11: Matriz de confusion (GT estricto)", "Figure 11: Confusion matrix (strict GT)"), w=120)

    # === 7. HALLAZGO CLAVE ===
    r.section(7, r.t("Hallazgo Clave: Dependencia Hidroelectrica", "Key Finding: Hydroelectric Dependency"))
    r.body(r.t(
        "El hallazgo mas importante del estudio: la efectividad del consenso CORRELACIONA con la "
        "dependencia hidroelectrica del pais. En paises con >30% de generacion hidro, el consenso "
        "funciona bien. En paises con <20% hidro, las crisis tienen firmas diferentes (precios de "
        "combustibles, olas de calor) que no son capturadas por STL/CUSUM.\n\n"
        "  Brasil (48% hidro): F1 = 0.727 - Crisis hidrica 2021\n"
        "  Ecuador (38% hidro): F1 = 0.750 - Sequia 2024\n"
        "  Colombia (41% hidro): F1 = 0.429 - El Nino 2024\n"
        "  Chile (14% hidro): F1 = 0.000 - Mega-sequia (no electrica)\n"
        "  Argentina (14% hidro): F1 = 0.000 - Ola de calor (termica)\n\n"
        "Esto no es una falla del modelo, es un hallazgo sobre el SCOPE DE APLICABILIDAD: el enfoque "
        "es efectivo para crisis hidricas en paises hidro-dependientes.",
        "The study's most important finding: consensus effectiveness CORRELATES with the country's "
        "hydroelectric dependency. In countries with >30% hydro generation, the consensus works well. "
        "In countries with <20% hydro, crises have different signatures (fuel prices, heatwaves) not "
        "captured by STL/CUSUM.\n\n"
        "  Brazil (48% hydro): F1 = 0.727 - 2021 water crisis\n"
        "  Ecuador (38% hydro): F1 = 0.750 - 2024 drought\n"
        "  Colombia (41% hydro): F1 = 0.429 - 2024 El Nino\n"
        "  Chile (14% hydro): F1 = 0.000 - Mega-drought (not electrical)\n"
        "  Argentina (14% hydro): F1 = 0.000 - Heatwave (thermal)\n\n"
        "This is not a model failure, but a finding about SCOPE OF APPLICABILITY: the approach is "
        "effective for hydro crises in hydro-dependent countries."
    ))
    r.img("29_hydro_vs_f1.png", r.t("Figura 12: Dependencia hidro vs F1", "Figure 12: Hydro dependency vs F1"))
    r.img("27_cross_country_heatmap.png", r.t("Figura 13: F1 por metodo y pais", "Figure 13: F1 by method and country"))

    # === 8. CROSS-COUNTRY ===
    r.section(8, r.t("Validacion Cross-Country", "Cross-Country Validation"))
    r.body(r.t(
        "El consenso se valida contra crisis documentadas oficialmente en 5 paises. Las fuentes "
        "oficiales incluyen decretos gubernamentales, informes de operadores de red y reportes "
        "de agencias reguladoras.",
        "The consensus is validated against officially documented crises in 5 countries. Official "
        "sources include government decrees, grid operator reports, and regulatory agency publications."
    ))
    r.img("20_cross_country.png", r.t("Figura 14: Consenso por pais", "Figure 14: Consensus by country"))
    r.img("28_consensus_wins.png", r.t("Figura 15: Consenso vs tecnicas individuales", "Figure 15: Consensus vs individual techniques"))
    r.img("04_heatmap_paises.png", r.t("Figura 16: Anomalias por pais y anio", "Figure 16: Anomalies by country and year"))

    # === 9. VALIDACION ESTADISTICA ===
    r.section(9, r.t("Validacion Estadistica", "Statistical Validation"))
    r.sub(r.t("Tests de hipotesis", "Hypothesis tests"))
    r.body(r.t(
        "Para confirmar que las anomalias detectadas representan comportamiento genuinamente diferente, "
        "se aplican tests formales (Mann-Whitney U, no parametrico) con tamano del efecto (Cohen's d):\n\n"
        "  gen_hydro:       normal=38.71%  anomalia=28.72%  p=0.0004***  d=2.81 (MUY GRANDE)\n"
        "  gen_fossil:      normal=11.87%  anomalia=21.47%  p=0.0004***  d=3.14 (MUY GRANDE)\n"
        "  co2_intensity:   normal=175.7   anomalia=298.5   p=0.0003***  d=3.15 (MUY GRANDE)\n"
        "  importaciones:   normal=0.04    anomalia=0.16    p=0.0045**   d=1.37 (GRANDE)\n"
        "  demanda_twh:     normal=2.86    anomalia=2.87    p=0.9702 ns  d=0.04 (NULO)\n\n"
        "Los Cohen's d >2.0 indican efectos MUY GRANDES. Los meses anomalos tienen perfiles "
        "dramaticamente diferentes en hidro/fosil/CO2. La demanda NO es diferente, lo cual tiene "
        "sentido: las crisis afectan el MIX de generacion, no la demanda total.",
        "To confirm detected anomalies represent genuinely different behavior, formal tests are applied "
        "(Mann-Whitney U, non-parametric) with effect size (Cohen's d):\n\n"
        "  gen_hydro:       normal=38.71%  anomaly=28.72%  p=0.0004***  d=2.81 (VERY LARGE)\n"
        "  gen_fossil:      normal=11.87%  anomaly=21.47%  p=0.0004***  d=3.14 (VERY LARGE)\n"
        "  co2_intensity:   normal=175.7   anomaly=298.5   p=0.0003***  d=3.15 (VERY LARGE)\n"
        "  imports:         normal=0.04    anomaly=0.16    p=0.0045**   d=1.37 (LARGE)\n"
        "  demanda_twh:     normal=2.86    anomaly=2.87    p=0.9702 ns  d=0.04 (NONE)\n\n"
        "Cohen's d >2.0 indicates VERY LARGE effects. Anomalous months have dramatically different "
        "hydro/fossil/CO2 profiles. Demand shows NO difference, which makes sense: crises affect the "
        "generation MIX, not total demand."
    ))
    r.img("14_boxplot_normal_vs_anomalia.png", r.t("Figura 17: Distribucion normal vs anomalia", "Figure 17: Normal vs anomaly distribution"))
    r.img("15_radar_perfil.png", r.t("Figura 18: Perfil comparativo", "Figure 18: Comparative profile"), w=140)

    r.sub("McNemar Test")
    r.body(r.t(
        "Test de McNemar comparando Consenso vs IF: p=1.000 (no significativo). Con N=73 meses y "
        "solo 3 meses de ground truth, NO hay poder estadistico suficiente para demostrar que el "
        "consenso es significativamente mejor que IF solo. La diferencia en F1 (0.750 vs 0.667) "
        "existe pero no alcanza significancia con esta muestra.",
        "McNemar test comparing Consensus vs IF: p=1.000 (not significant). With N=73 months and "
        "only 3 months of ground truth, there is NOT sufficient statistical power to demonstrate that "
        "the consensus is significantly better than IF alone. The F1 difference (0.750 vs 0.667) "
        "exists but does not reach significance with this sample."
    ))

    # === 10. CIs ===
    r.section(10, r.t("Intervalos de Confianza", "Confidence Intervals"))
    r.body(r.t(
        "Bootstrap con 1000 iteraciones, 95% de confianza:\n\n"
        "  Consenso:  F1 = 0.748  IC 95% = [0.400, 1.000]\n"
        "  IF:        F1 = 0.664  IC 95% = [0.286, 1.000]\n"
        "  STL:       F1 = 0.380  IC 95% = [0.000, 0.750]\n"
        "  CUSUM:     F1 = 0.318  IC 95% = [0.105, 0.571]\n\n"
        "Los intervalos son AMPLIOS, reflejando la muestra pequena (N=73). Los CIs de Consenso "
        "e IF se SOLAPAN, lo que es consistente con el McNemar no significativo. Esto NO invalida "
        "el modelo, pero limita la fuerza de las conclusiones.",
        "Bootstrap with 1000 iterations, 95% confidence:\n\n"
        "  Consensus: F1 = 0.748  95% CI = [0.400, 1.000]\n"
        "  IF:        F1 = 0.664  95% CI = [0.286, 1.000]\n"
        "  STL:       F1 = 0.380  95% CI = [0.000, 0.750]\n"
        "  CUSUM:     F1 = 0.318  95% CI = [0.105, 0.571]\n\n"
        "Intervals are WIDE, reflecting the small sample (N=73). Consensus and IF CIs OVERLAP, "
        "consistent with the non-significant McNemar test. This does NOT invalidate the model but "
        "limits the strength of conclusions."
    ))
    r.img("21_confidence_intervals.png", r.t("Figura 19: Intervalos de confianza", "Figure 19: Confidence intervals"))

    # === 11. SENSITIVITY ===
    r.section(11, r.t("Analisis de Sensibilidad", "Sensitivity Analysis"))
    r.body(r.t(
        "Se evaluan 36 combinaciones de parametros (contamination x stl_sigma x cusum_factor) "
        "para determinar la robustez del modelo ante cambios de configuracion.\n\n"
        "Mejor configuracion encontrada:\n"
        "  contamination=0.05, stl_sigma=1.5, cusum_factor=4.0\n"
        "  F1=0.857, MCC=0.861\n\n"
        "La configuracion por defecto (contamination=0.08, stl=2.0, cusum=4.0) produce F1=0.750, "
        "lo que sugiere margen de mejora con tuning mas agresivo pero con riesgo de sobreajuste.",
        "36 parameter combinations (contamination x stl_sigma x cusum_factor) are evaluated "
        "to determine model robustness to configuration changes.\n\n"
        "Best configuration found:\n"
        "  contamination=0.05, stl_sigma=1.5, cusum_factor=4.0\n"
        "  F1=0.857, MCC=0.861\n\n"
        "The default configuration (contamination=0.08, stl=2.0, cusum=4.0) produces F1=0.750, "
        "suggesting room for improvement with more aggressive tuning but with overfitting risk."
    ))
    r.img("22_sensitivity_heatmap.png", r.t("Figura 20: Heatmap de sensibilidad", "Figure 20: Sensitivity heatmap"))
    r.img("26_sensitivity_scatter.png", r.t("Figura 21: Scatter de sensibilidad", "Figure 21: Sensitivity scatter"))

    # === 12. STL ===
    r.section(12, r.t("Descomposicion STL Detallada", "STL Decomposition Detail"))
    r.body(r.t(
        "La descomposicion STL de la generacion hidroelectrica de Ecuador revela:\n\n"
        "TENDENCIA: Declive gradual de ~45% (2019) a ~35% (2025), reflejando crecimiento de demanda "
        "sin nueva capacidad hidro instalada.\n\n"
        "ESTACIONALIDAD: Patron anual claro - mayor hidro en meses lluviosos (octubre-mayo), menor "
        "en estacion seca de la Sierra (junio-septiembre).\n\n"
        "RESIDUALES EXTREMOS:\n"
        "  Nov 2024: -13.8 (-4.8 sigma) - el mas extremo de toda la serie\n"
        "  Oct 2024: -10.9 (-3.8 sigma)\n"
        "  Sep 2024: -7.4 (-2.6 sigma)\n"
        "  Oct/Nov 2023: ~-6.5 (-2.2 sigma) - detecto la sequia de 2023\n\n"
        "STL es la UNICA tecnica que detecto parcialmente la crisis gradual de 2023.",
        "STL decomposition of Ecuador's hydroelectric generation reveals:\n\n"
        "TREND: Gradual decline from ~45% (2019) to ~35% (2025), reflecting demand growth without "
        "new hydro capacity.\n\n"
        "SEASONALITY: Clear annual pattern - higher hydro in rainy months (October-May), lower in "
        "Sierra dry season (June-September).\n\n"
        "EXTREME RESIDUALS:\n"
        "  Nov 2024: -13.8 (-4.8 sigma) - most extreme in the entire series\n"
        "  Oct 2024: -10.9 (-3.8 sigma)\n"
        "  Sep 2024: -7.4 (-2.6 sigma)\n"
        "  Oct/Nov 2023: ~-6.5 (-2.2 sigma) - captured the 2023 drought\n\n"
        "STL is the ONLY technique that partially detected the gradual 2023 crisis."
    ))
    r.img("10_stl_decomposition.png", r.t("Figura 22: Descomposicion STL completa", "Figure 22: Full STL decomposition"))

    # === 13. MULTI-PAIS ===
    r.section(13, r.t("Analisis Multi-Pais", "Multi-Country Analysis"))
    r.img("13_timeline_latam.png", r.t("Figura 23: Timeline de anomalias 8 paises", "Figure 23: Anomaly timeline 8 countries"))
    r.img("16_latam_hidro_anomalias.png", r.t("Figura 24: Hidro + anomalias por pais", "Figure 24: Hydro + anomalies by country"))
    r.img("09_latam_hidro_comparativo.png", r.t("Figura 25: Comparativo hidro regional", "Figure 25: Regional hydro comparison"))

    # === 14. LIMITACIONES ===
    r.section(14, r.t("Limitaciones", "Limitations"))
    r.body(r.t(
        "1. McNEMAR p=1.0: No se puede probar estadisticamente que consenso > IF con N=73. La diferencia "
        "F1 (0.750 vs 0.667) existe pero N no da poder estadistico suficiente.\n\n"
        "2. BOOTSTRAP CI AMPLIO: F1=[0.400, 1.000]. Refleja la muestra pequena, no falla del modelo.\n\n"
        "3. FALLA EN PAISES FOSILES: Chile (F1=0.0) y Argentina (F1=0.0) tienen crisis no hidricas "
        "(precios, calor) que requieren variables diferentes.\n\n"
        "4. GT AMPLIO RECALL=33%: El modelo detecta los 3 meses pico de 9. Los 6 meses de crisis "
        "moderada (abr-sep) se diluyen en datos mensuales.\n\n"
        "5. N LIMITADO: 85 meses Ecuador, 3 meses GT. Cualquier analisis con N<100 tiene limitaciones "
        "inherentes en significancia estadistica.\n\n"
        "6. DATOS MENSUALES: Granularidad mensual suaviza crisis graduales y no detecta apagones "
        "programados (que no cambian volumenes de generacion).\n\n"
        "7. CONSENSO PONDERADO = SIMPLE: Para Ecuador y Brasil, ambos dan los mismos resultados. "
        "La contribucion teorica no se manifiesta numericamente en estos datos.",
        "1. McNEMAR p=1.0: Cannot statistically prove consensus > IF with N=73. F1 difference "
        "(0.750 vs 0.667) exists but insufficient statistical power.\n\n"
        "2. WIDE BOOTSTRAP CI: F1=[0.400, 1.000]. Reflects small sample, not model failure.\n\n"
        "3. FAILS IN FOSSIL COUNTRIES: Chile (F1=0.0) and Argentina (F1=0.0) have non-hydro crises "
        "(prices, heat) requiring different variables.\n\n"
        "4. BROAD GT RECALL=33%: Model detects 3 peak months out of 9. The 6 moderate crisis months "
        "(Apr-Sep) are smoothed in monthly data.\n\n"
        "5. LIMITED N: 85 months Ecuador, 3 months GT. Any analysis with N<100 has inherent "
        "limitations in statistical significance.\n\n"
        "6. MONTHLY DATA: Monthly granularity smooths gradual crises and cannot detect programmed "
        "blackouts (which don't change generation volumes).\n\n"
        "7. WEIGHTED CONSENSUS = SIMPLE: For Ecuador and Brazil, both give identical results. "
        "The theoretical contribution does not manifest numerically in this data."
    ))

    # === 15. FUENTES OFICIALES ===
    r.section(15, r.t("Fuentes Oficiales del Ground Truth", "Official Ground Truth Sources"))
    r.body(r.t(
        "ECUADOR:\n"
        "  Decreto Ejecutivo No. 229 (19-abr-2024): Estado de excepcion\n"
        "  Acuerdo Ministerial MEM-MEM-2024-0005-AM: Emergencia sector electrico\n"
        "  CENACE Informe Anual 2024\n"
        "  Impacto: peor sequia en 61 anios, 1.5% PIB\n\n"
        "BRASIL:\n"
        "  Decreto 10.939/2021: Operaciones financieras distribuidoras\n"
        "  Medida Provisoria 1.055/2021: Creacion CREG\n"
        "  EPE NT-DEE-DEA-001-2023: Diagnostico escasez hidrica\n"
        "  Nature d41586-021-03625-w: Brazil water crisis\n\n"
        "COLOMBIA:\n"
        "  XM Colombia: Informes variables del mercado abr-2024\n"
        "  Precios mayoristas: +22.68% en abril 2024\n\n"
        "CHILE:\n"
        "  Universidad de Chile: Analisis mega-sequia\n"
        "  Biblioteca del Congreso Nacional: Informacion territorial\n\n"
        "ARGENTINA:\n"
        "  SMN Argentina: Informe ola de calor 2022\n"
        "  FARN: Impactos cambio climatico en sistema energetico",
        "ECUADOR:\n"
        "  Executive Decree No. 229 (Apr 19, 2024): State of exception\n"
        "  Ministerial Accord MEM-MEM-2024-0005-AM: Electricity sector emergency\n"
        "  CENACE Annual Report 2024\n"
        "  Impact: worst drought in 61 years, 1.5% GDP\n\n"
        "BRAZIL:\n"
        "  Decree 10.939/2021: Financial operations for distributors\n"
        "  Provisional Measure 1.055/2021: CREG creation\n"
        "  EPE NT-DEE-DEA-001-2023: Water scarcity diagnosis\n"
        "  Nature d41586-021-03625-w: Brazil water crisis\n\n"
        "COLOMBIA:\n"
        "  XM Colombia: Market variables reports Apr 2024\n"
        "  Wholesale prices: +22.68% in April 2024\n\n"
        "CHILE:\n"
        "  Universidad de Chile: Mega-drought analysis\n"
        "  National Congress Library: Territorial information\n\n"
        "ARGENTINA:\n"
        "  SMN Argentina: 2022 heatwave report\n"
        "  FARN: Climate change impacts on energy system"
    ))

    # === 16. APLICACIONES ===
    r.section(16, r.t("Aplicaciones Potenciales", "Potential Applications"))
    r.body(r.t(
        "SECTOR ENERGETICO:\n"
        "  - Operadores de red (CENACE, XM, COES, ONS): Alerta temprana de crisis\n"
        "  - Reguladores (ARCERNNR): Monitoreo automatizado del sector\n"
        "  - Traders: Senales basadas en anomalias para prediccion de precios\n\n"
        "CLIMA Y MEDIO AMBIENTE:\n"
        "  - Evaluacion de impacto de El Nino en generacion hidroelectrica\n"
        "  - Monitoreo de intensidad de carbono para reportes ESG\n\n"
        "ACADEMIA:\n"
        "  - Estudios comparativos de seguridad energetica regional\n"
        "  - Metodologia multi-tecnica para deteccion de anomalias\n"
        "  - Consenso adaptativo como contribucion novel\n\n"
        "PUBLICACION SUGERIDA:\n"
        '  Titulo: "Hydro-Weighted Consensus Anomaly Detection for Energy\n'
        '  Crises in Latin American Power Grids"\n'
        "  Venues: Energy Policy, Applied Energy, IEEE Access, Energies (MDPI)",
        "ENERGY SECTOR:\n"
        "  - Grid operators (CENACE, XM, COES, ONS): Crisis early warning\n"
        "  - Regulators (ARCERNNR): Automated sector monitoring\n"
        "  - Traders: Anomaly-based signals for price prediction\n\n"
        "CLIMATE:\n"
        "  - El Nino impact assessment on hydroelectric generation\n"
        "  - Carbon intensity monitoring for ESG reporting\n\n"
        "ACADEMIC:\n"
        "  - Regional energy security comparative studies\n"
        "  - Multi-technique anomaly detection methodology\n"
        "  - Adaptive consensus as novel contribution\n\n"
        "SUGGESTED PUBLICATION:\n"
        '  Title: "Hydro-Weighted Consensus Anomaly Detection for Energy\n'
        '  Crises in Latin American Power Grids"\n'
        "  Venues: Energy Policy, Applied Energy, IEEE Access, Energies (MDPI)"
    ))

    # === 17. REFERENCIAS ===
    r.section(17, r.t("Referencias", "References"))
    r.body(
        "[1] Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). Isolation Forest. "
        "Proc. IEEE International Conference on Data Mining (ICDM).\n\n"
        "[2] Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. (1990). "
        "STL: A Seasonal-Trend Decomposition Procedure Based on LOESS. "
        "Journal of Official Statistics, 6(1), 3-73.\n\n"
        "[3] Page, E.S. (1954). Continuous Inspection Schemes. Biometrika, 41(1/2), 100-115.\n\n"
        "[4] Chandola, V., Banerjee, A., Kumar, V. (2009). Anomaly Detection: A Survey. "
        "ACM Computing Surveys, 41(3), 1-58.\n\n"
        "[5] Himeur, Y., et al. (2021). Artificial intelligence based anomaly detection "
        "of energy consumption in buildings: A review. Applied Energy, 286, 116601.\n\n"
        "[6] Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter "
        "Optimization Framework. Proc. ACM SIGKDD.\n\n"
        "[7] CENACE (2025). Informe Anual 2024. cenace.gob.ec\n\n"
        "[8] Ember (2026). Global Electricity Data. ember-energy.org\n\n"
        "[9] Our World in Data (2026). Energy Dataset. ourworldindata.org\n\n"
        "[10] XM Colombia (2024). Market Variables Reports. xm.com.co\n\n"
        "[11] EPE Brazil (2023). Water Scarcity Diagnosis NT-DEE-DEA-001-2023."
    )

    return r


if __name__ == "__main__":
    for lang in ["es", "en"]:
        r = build_report(lang)
        suffix = "Reporte_ES" if lang == "es" else "Report"
        out = ROOT / "docs" / f"Ecuador_Energy_Anomalies_{suffix}.pdf"
        r.output(str(out))
        print(f"{lang.upper()}: {out} ({r.page_no()} pages)")
