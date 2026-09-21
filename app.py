import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Predictor de Renuncia Voluntaria", layout="centered")

# Carga del modelo
with open('modelo_atricion.pkl', 'rb') as f:
    modelo = pickle.load(f)

st.title("Predictor de Renuncia Voluntaria")
st.write(
    "Complete los campos y presione el boton para obtener la prediccion "
    "de riesgo de renuncia voluntaria del empleado."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    overtime = st.selectbox("Realiza horas extra", options=[0, 1],
                            format_func=lambda x: "Si" if x == 1 else "No")
    total_working_years = st.slider("Años de experiencia total", 0, 40, 5)
    job_level = st.selectbox("Nivel jerarquico", options=[1, 2, 3, 4, 5],
                             format_func=lambda x: {1:"1 - Junior", 2:"2 - Semi-senior",
                             3:"3 - Senior", 4:"4 - Gerencia", 5:"5 - Alta Direccion"}[x])
    age = st.slider("Edad", 18, 65, 30)
    monthly_income = st.number_input("Salario mensual (USD)", 1000, 20000, 5000, 500)

with col2:
    job_satisfaction = st.selectbox("Satisfaccion con el trabajo", options=[1,2,3,4],
                                    format_func=lambda x: {1:"1 - Baja", 2:"2 - Media",
                                    3:"3 - Alta", 4:"4 - Muy alta"}[x])
    environment_satisfaction = st.selectbox("Satisfaccion con el ambiente", options=[1,2,3,4],
                                            format_func=lambda x: {1:"1 - Baja", 2:"2 - Media",
                                            3:"3 - Alta", 4:"4 - Muy alta"}[x])
    work_life_balance = st.selectbox("Balance vida-trabajo", options=[1,2,3,4],
                                     format_func=lambda x: {1:"1 - Malo", 2:"2 - Regular",
                                     3:"3 - Bueno", 4:"4 - Excelente"}[x])
    years_at_company = st.slider("Años en la empresa", 0, 40, 3)
    years_with_manager = st.slider("Años con el jefe actual", 0, 20, 2)

st.divider()

if st.button("Obtener prediccion", use_container_width=True):
    entrada = np.array([[overtime, total_working_years, job_level, age,
                         monthly_income, job_satisfaction, environment_satisfaction,
                         work_life_balance, years_at_company, years_with_manager]])

    prediccion   = modelo.predict(entrada)[0]
    probabilidad = modelo.predict_proba(entrada)[0]

    st.divider()
    st.subheader("Resultado")

    if prediccion == 1:
        st.error("El empleado presenta ALTO riesgo de renuncia voluntaria.")
    else:
        st.success("El empleado presenta BAJO riesgo de renuncia voluntaria.")

    st.write(f"Probabilidad de renuncia:     **{probabilidad[1]*100:.1f}%**")
    st.write(f"Probabilidad de permanencia:  **{probabilidad[0]*100:.1f}%**")

    st.caption(
        "Este resultado es orientativo y no debe usarse como unico criterio "
        "para decisiones sobre el empleado."
    )
