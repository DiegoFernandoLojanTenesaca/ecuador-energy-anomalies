# Post LinkedIn — Ecuador Energy Anomaly Detector

---

Hace unos meses me puse a trabajar en algo que me parecía un problema interesante:
¿se puede detectar automáticamente una crisis eléctrica antes de que sea evidente?

El contexto era Ecuador. El país genera ~70% de su electricidad con hidroeléctricas,
lo que lo hace muy vulnerable a sequías. En 2024 hubo una severa que derivó en
apagones de hasta 14 horas diarias. Quería ver si eso era detectable con datos públicos
y sin etiquetas manuales.

Usé datos mensuales de Ember para 8 países de LATAM (2018–2025) y combiné
tres técnicas: Isolation Forest para outliers multivariados, STL para detectar
residuos anómalos en la serie de tiempo, y CUSUM para cambios estructurales en
generación hidro. Un mes se marca como anómalo solo si al menos dos de las tres
técnicas coinciden — eso baja bastante la tasa de falsos positivos.

Comparé los resultados contra 9 modelos y los validé con crisis documentadas
en decretos oficiales de Ecuador, Brasil y Colombia.

Sobre el período más severo de la crisis ecuatoriana (oct–dic 2024):
F1 = 0.750, MCC = 0.765, los tres meses detectados.

---

Lo que no anticipé fue la relación entre dependencia hidro y efectividad del modelo.

Ecuador  (38% hidro) → F1 = 0.750
Brasil   (48% hidro) → F1 = 0.727
Colombia (41% hidro) → F1 = 0.429
Chile    (14% hidro) → F1 = 0.000
Argentina(14% hidro) → F1 = 0.000

Tiene lógica una vez que lo ves: STL y CUSUM están leyendo la curva de generación
hidroeléctrica. Donde esa curva no domina el mix energético, no hay señal.
Las crisis de Chile y Argentina son térmicas — demanda por calor, no sequía —
y eso el modelo simplemente no lo captura.

Es una limitación honesta, pero también dice algo sobre cómo diseñar
detección de anomalías en energía según el perfil del país.

---

Los datos los obtuve scrapeando directamente desde Ember (CC BY 4.0),
sin datasets prearmados. El repo lo voy a dejar público en los próximos días.

Si trabajas con series temporales, energía o detección de anomalías,
me interesa saber qué piensas del enfoque.

🔗 Demo: [link]
📁 Repo: próximamente en GitHub

#MachineLearning #DataScience #AnomalyDetection #OpenData #Ecuador #EnergySector #Python
