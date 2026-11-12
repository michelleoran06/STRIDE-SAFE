# STRIDE-SAFE — interfaz gráfica con Streamlit
# Para abrirla, en la terminal (dentro de la carpeta del proyecto):  streamlit run app.py
import joblib
import pandas as pd
import streamlit as st

CARPETA = 'data/'   # carpeta donde están el modelo y la tabla que guardó el notebook (sección 4)
PISO_DE = 0.5       # desviación mínima de la etiqueta (D-07), igual que en el notebook

# Variables de bienestar: (texto en pantalla, mínimo, máximo, paso del slider)
ESCALAS = {
    'readiness':      ('Readiness (1 = not ready, 10 = fully ready)', 1, 10, 1.0),
    'fatigue':        ('Fatigue (1 = very tired, 5 = very fresh)', 1, 5, 1.0),
    'mood':           ('Mood (1 = very bad, 5 = very good)', 1, 5, 1.0),
    'sleep_duration': ('Sleep duration (hours)', 0, 12, 0.5),
    'sleep_quality':  ('Sleep quality (1 = very bad, 5 = very good)', 1, 5, 1.0),
    'soreness':       ('Muscle soreness (1 = very sore, 5 = not sore)', 1, 5, 1.0),
    'stress':         ('Stress (1 = very stressed, 5 = very relaxed)', 1, 5, 1.0),
}


# Cargamos el modelo y la tabla una sola vez (cache = no se recargan en cada clic)
@st.cache_resource
def cargar_modelo_final():   # nombre nuevo para que Streamlit no reutilice la caché del modelo anterior
    paquete = joblib.load(CARPETA + 'modelo_stride_safe.joblib')
    tabla = pd.read_csv(CARPETA + 'tabla_variables.csv', parse_dates=['fecha'])
    tabla = tabla[tabla['año'] == 2021].copy()   # solo 2021: días que el modelo NO vio al entrenar
    # Umbral personal de caída: media de 28 días - max(desviación, 0.5), igual que la etiqueta
    tabla['limite_caida'] = tabla['media_28'] - tabla['de_28'].clip(lower=PISO_DE)
    return paquete['modelo'], paquete['calibrador'], paquete['umbral'], paquete['variables'], tabla


modelo, calibrador, UMBRAL, VARIABLES, tabla = cargar_modelo_final()
# El umbral se eligió en 2020 sobre la salida del bosque; aquí lo pasamos a probabilidad calibrada
UMBRAL_CAL = float(calibrador.predict([UMBRAL])[0])

st.title('STRIDE-SAFE')
st.caption("Probability that a player's readiness drops tomorrow below her usual level")

# ---------- 1. Elegir jugadora y día (barra lateral) ----------
st.sidebar.header('Player and day')
jugadora = st.sidebar.selectbox('Player', sorted(tabla['jugadora_id'].unique()),
                                format_func=lambda j: j[:14])        # mostramos solo 'TeamA-xxxxxxxx'
datos_j = tabla[tabla['jugadora_id'] == jugadora]
fecha = st.sidebar.selectbox('Day', datos_j['fecha'].dt.date.tolist()[::-1])   # más reciente primero
fila = datos_j[datos_j['fecha'].dt.date == fecha].iloc[0]                    # el registro de ese día

# ---------- 2. Datos de hoy (se pueden modificar) ----------
st.subheader("Today's data")
st.write('Values come from the record of that day. Change them to see how the risk changes.')
izq, der = st.columns(2)
carga = izq.number_input('Training load today (RPE × minutes)', 0, 4000, int(fila['daily_load']), step=10)
respuestas = {}
for col, (texto, minimo, maximo, paso) in ESCALAS.items():
    respuestas[col] = der.slider(texto, float(minimo), float(maximo), float(fila[col]), step=paso)

# ---------- 3. Armar las variables que espera el modelo ----------
x = fila[VARIABLES].astype(float).copy()

# Si cambia la carga de hoy, el promedio de 7 días (ATL) y el de 28 días (CTL28) cambian en proporción
cambio = carga - fila['daily_load']
x['daily_load'] = carga
x['atl'] = fila['atl'] + cambio / 7
x['ctl28'] = fila['ctl28'] + cambio / 28

# Bienestar: valor nuevo y su diferencia contra el promedio personal (promedio = valor - diferencia)
for col in ESCALAS:
    promedio = fila[col] - fila[col + '_dif']
    x[col] = respuestas[col]
    x[col + '_dif'] = respuestas[col] - promedio

# Puntaje del bosque (columna 1 = clase positiva) y su probabilidad calibrada
puntaje = modelo.predict_proba(pd.DataFrame([x]))[0, 1]
prob = float(calibrador.predict([puntaje])[0])

# ---------- 4. Resultado ----------
st.subheader('Result')
st.metric('Probability of readiness decline tomorrow', f'{prob:.0%}')
if puntaje >= UMBRAL:   # misma decisión que en el notebook (3.8)
    st.error('ALERT: high risk of readiness decline tomorrow. Consider reducing load or scheduling rest.')
else:
    st.success('No alert: readiness is expected to stay within her usual range.')
st.caption(f'Alert from {UMBRAL_CAL:.0%} (threshold chosen with 2020 data). '
           f'On average about 15 % of days end in a decline, so 20–30 % is already high risk.')

# Lo que pasó en realidad (solo porque son datos históricos)
limite = fila['limite_caida']
st.write(f"**What actually happened:** readiness tomorrow = {fila['readiness_manana']:.0f} "
         f"(her decline threshold = {limite:.1f}) → {'decline' if fila['target'] == 1 else 'no decline'}")

# ---------- 5. Historia de la jugadora ----------
st.subheader('Readiness history (last 28 reports)')
historia = datos_j[datos_j['fecha'].dt.date <= fecha].tail(28).set_index('fecha')
historia['decline threshold'] = historia['limite_caida']
st.line_chart(historia[['readiness', 'decline threshold']])

# ---------- 6. Variables que más pesan en el modelo ----------
st.subheader('Most important variables (Random Forest)')
importancia = pd.Series(modelo.feature_importances_, index=VARIABLES).sort_values(ascending=False).head(8)
st.bar_chart(importancia, horizontal=True, sort=False)   # sort=False respeta el orden de mayor a menor

# ---------- 7. Advertencia de uso ----------
st.warning('STRIDE-SAFE is a decision-support tool, not a medical diagnosis. It was trained with '
           'self-reported data from two Norwegian elite teams (SoccerMon dataset). '
           'Final decisions belong to the medical and coaching staff.')
