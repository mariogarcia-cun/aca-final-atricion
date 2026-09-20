import streamlit as st
import pickle
import numpy as np

# Configuracion de la pagina
st.set_page_config(
    page_title="Predictor de Renuncia Voluntaria",
    page_icon=None,
    layout="centered"
)

# Carga del modelo
@st.cache_resource
def cargar_modelo():
    with open('modelo_atricion.pkl', 'rb') as f:
        return pickle.load(f)

modelo = cargar_modelo()

# Titulo y descripcion
st.title("Predictor de Renuncia Voluntaria")
st.write(
    "Esta herramienta permite estimar si un empleado tiene riesgo de renunciar "
    "voluntariamente, a partir de sus caracteristicas laborales. "
    "Complete los campos y presione el boton para obtener la prediccion."
)

st.divider()

# Formulario de entrada
st.subheader("Datos del empleado")

col1, col2 = st.columns(2)

with col1:
    overtime = st.selectbox(
        "Realiza horas extra",
        options=[0, 1],
        format_func=lambda x: "Si" if x == 1 else "No"
    )
    total_working_years = st.slider(
        "Anos de experiencia laboral total",
        min_value=0, max_value=40, value=5
    )
    job_level = st.selectbox(
        "Nivel jerarquico",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "1 - Junior",
            2: "2 - Semi-senior",
            3: "3 - Senior",
            4: "4 - Gerencia",
            5: "5 - Alta Direccion"
        }[x]
    )
    age = st.slider(
        "Edad del empleado",
        min_value=18, max_value=65, value=30
    )
    monthly_income = st.number_input(
        "Salario mensual (USD)",
        min_value=1000, max_value=20000, value=5000, step=500
    )

with col2:
    job_satisfaction = st.selectbox(
        "Satisfaccion con el trabajo",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Baja",
            2: "2 - Media",
            3: "3 - Alta",
            4: "4 - Muy alta"
        }[x]
    )
    environment_satisfaction = st.selectbox(
        "Satisfaccion con el ambiente",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Baja",
            2: "2 - Media",
            3: "3 - Alta",
            4: "4 - Muy alta"
        }[x]
    )
    work_life_balance = st.selectbox(
        "Balance vida-trabajo",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Malo",
            2: "2 - Regular",
            3: "3 - Bueno",
            4: "4 - Excelente"
        }[x]
    )
    years_at_company = st.slider(
        "Anos en la empresa",
        min_value=0, max_value=40, value=3
    )
    years_with_manager = st.slider(
        "Anos con el jefe actual",
        min_value=0, max_value=20, value=2
    )

st.divider()

# Prediccion
if st.button("Obtener prediccion", use_container_width=True):

    entrada = np.array([[
        overtime,
        total_working_years,
        job_level,
        age,
        monthly_income,
        job_satisfaction,
        environment_satisfaction,
        work_life_balance,
        years_at_company,
        years_with_manager
    ]])

    prediccion = modelo.predict(entrada)[0]
    probabilidad = modelo.predict_proba(entrada)[0]

    st.divider()
    st.subheader("Resultado")

    if prediccion == 1:
        st.error("El empleado presenta ALTO riesgo de renuncia voluntaria.")
    else:
        st.success("El empleado presenta BAJO riesgo de renuncia voluntaria.")

    st.write(f"Probabilidad de renuncia:     **{probabilidad[1]*100:.1f}%**")
    st.write(f"Probabilidad de permanencia:  **{probabilidad[0]*100:.1f}%**")

    st.divider()
    st.caption(
        "Este resultado es orientativo. La decision final debe complementarse "
        "con el criterio del area de recursos humanos. El modelo no debe usarse "
        "de forma punitiva ni como unico criterio para decisiones sobre el empleado."
    )
