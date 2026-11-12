# STRIDE-SAFE

Interfaz (Streamlit) que estima la probabilidad de que la disposición para entrenar (readiness) de una jugadora caiga mañana por debajo de su nivel habitual.

- Datos: SoccerMon (Midoglu et al., 2024), dos equipos noruegos de fútbol femenil de élite, 2020–2021.
- Modelo: Random Forest entrenado con 2020; la app muestra solo días de 2021.
- La probabilidad está calibrada (regresión isotónica con predicciones fuera de pliegue de 2020) y la alerta usa el umbral elegido en 2020 (sección 3.8 del notebook). Los archivos de `data/` los genera la sección 4 del notebook.
- Herramienta de apoyo a decisiones, no diagnóstico médico.

Para correrla en local:

```
pip install -r requirements.txt
streamlit run app.py
```

Proyecto de Machine Learning — Gonzalez Becerril Michel y Lara Orán Fernanda Michelle.
