# AI DIABETES CLINICAL ASSISTANT

# IMPORTACIONES

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import ollama

from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer)
from reportlab.lib.styles import (getSampleStyleSheet)

# CONFIGURACIÓN PÁGINA

st.set_page_config(page_title="AI Diabetes Clinical Assistant",page_icon="🩺",layout="wide")

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.metric-container{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# CARGA DEL SISTEMA

@st.cache_resource
def load_system():

    model = joblib.load( "modelo_diabetes_readmission.pkl")
    threshold = joblib.load("best_threshold.pkl")
    features = joblib.load("features.pkl")
    cat_features = joblib.load("cat_features.pkl")

    return model, threshold, features, cat_features

best_model, best_threshold, features, cat_features = load_system()

# CREAMOS EL EXPLAINER SHAP

explainer = shap.TreeExplainer(best_model)

# DICCIONARIO DE NOMBRES DE VARIABLES PARA INTERPRETACIÓN

variable_names = {
    "number_inpatient": "Ingresos hospitalarios previos",
    "previous_utilization": "Utilización previa del sistema sanitario",
    "high_risk_history": "Historial clínico de riesgo",
    "multiple_admissions": "Múltiples ingresos previos",
    "num_medications":"Número de medicamentos",
    "number_diagnoses":"Número de diagnósticos",
    "time_in_hospital":"Duración de la hospitalización",
    "clinical_complexity":"Complejidad clínica",
    "hospital_intensity":"Intensidad asistencial",
    "medication_load":"Carga farmacológica",
    "A1Cresult": "Resultado HbA1c",
    "diag_1_group": "Diagnóstico principal",
    "diag_2_group":"Diagnóstico secundario",
    "diag_3_group": "Diagnóstico terciario"
}

# CABECERA

st.title("🩺 AI Diabetes Clinical Assistant")

st.markdown("""
### Sistema Inteligente de Apoyo a la Decisión Clínica

Predicción de riesgo de reingreso hospitalario en pacientes con diabetes mediante Inteligencia Artificial.
""")

st.divider()

# SIDEBAR

st.sidebar.title("⚙️ Sistema")
st.sidebar.success("Modelo cargado correctamente")
st.sidebar.write(f"Threshold óptimo: {best_threshold:.2f}")
st.sidebar.write( f"Número variables: {len(features)}")

# PESTAÑAS

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🩺 Predicción",
        "📄 Informe clínico",
        "🤖 AI Assistant",
        "ℹ️ Sobre el modelo"
    ]
)

# FUNCIÓN INFORME CLÍNICO

def build_clinical_report(risk,label,recommendations,top_factors):

    report = f"""
AI DIABETES CLINICAL ASSISTANT

==================================================

PROBABILIDAD DE REINGRESO

{risk*100:.2f} %

CLASIFICACIÓN

{label}

==================================================

FACTORES PRINCIPALES

"""

    for _, row in top_factors.iterrows():

        report += (f"\n• {row['variable']}")

    report += "\n\nRECOMENDACIONES\n"

    for rec in recommendations:

        report += (f"\n• {rec}")

    return report

# FUNCIÓN REPORTE EN PDF 

def create_pdf_report(clinical_report):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    for line in clinical_report.split("\n"):

        elements.append(Paragraph(line,styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return buffer

# FUNCIÓN LLM CLINICAL ASSISTANT

def ask_ai_assistant(question,risk,label,top_factors,recommendations,patient_df):
    factors_text = ""

    for _, row in top_factors.iterrows():

        factor = variable_names.get(row["variable"],row["variable"])
        direction = (
        "incrementa el riesgo"
            if row["shap_value"] > 0
            else "reduce el riesgo"
        )

        factors_text += (f"- {factor}: {direction} "f"(SHAP={row['shap_value']:.3f})\n")

    recommendations_text = "\n".join(recommendations)
    age = patient_df["age"].iloc[0]

    num_medications = patient_df["num_medications"].iloc[0]
    number_diagnoses = patient_df["number_diagnoses"].iloc[0]
    number_inpatient = patient_df["number_inpatient"].iloc[0]
    number_outpatient = patient_df["number_outpatient"].iloc[0]
    number_emergency = patient_df["number_emergency"].iloc[0]
    time_in_hospital = patient_df["time_in_hospital"].iloc[0]
    a1c = patient_df["A1Cresult"].iloc[0]
    max_glu = patient_df["max_glu_serum"].iloc[0]


    prompt = f"""
    Eres un asistente clínico especializado en diabetes.

    Información del paciente:

    Edad:{age}
    Días hospitalizado:{time_in_hospital}
    Número de medicamentos:{num_medications}
    Número de diagnósticos:{number_diagnoses}
    Ingresos hospitalarios previos:{number_inpatient}
    Consultas externas:{number_outpatient}
    Urgencias previas:{number_emergency}
    Resultado HbA1c:{a1c}
    Glucosa máxima:{max_glu}

    Probabilidad estimada de reingreso:{risk*100:.2f}%
    Clasificación calculada por el modelo:{label}
    Factores principales identificados por SHAP:{factors_text}
    Recomendaciones actuales:{recommendations_text}
    Pregunta del usuario:{question}

    IMPORTANTE:
    
    La clasificación oficial es la calculada por el modelo.
    No la modifiques.
    No recalcules el riesgo.
    Utiliza siempre la clasificación proporcionada.
    Explica también el significado de los impactos SHAP.
    Responde en español.
    Sé profesional.
    Sé breve.
    No inventes diagnósticos.
    No sustituyas al médico.

    Estructura la respuesta en:

    1. Interpretación
    2. Factores relevantes
    3. Recomendaciones

    Máximo 150 palabras.
    """

    response = ollama.chat(model="llama3.2:3b",messages=[ {"role":"user","content":prompt}])
    return response["message"]["content"]


# FORMULARIO

with tab1:

    st.header("📋 Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Edad",["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"],index=7)
    gender = st.selectbox("Sexo",["Male", "Female"])
    race = st.selectbox("Raza",[ "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    time_in_hospital = st.slider("Días hospitalizado",1,14,7)
    number_diagnoses = st.slider("Número diagnósticos",1,16,9)
    num_medications = st.slider("Número medicamentos",1,80,24)
    num_lab_procedures = st.slider( "Pruebas laboratorio", 1, 132, 65)

with col2:

    number_inpatient = st.slider("Ingresos previos",0,20,3)
    number_outpatient = st.slider("Consultas externas",0,40,2)
    number_emergency = st.slider("Urgencias previas",0,20,1)
    a1c = st.selectbox("Resultado HbA1c",["None","Norm",">7",">8"])
    max_glu = st.selectbox("Glucosa máxima",["None","Norm",">200",">300"])

predict_button = st.button("🔍 Predecir riesgo")

# FUNCIÓN RECOMENDACIONES CLÍNICAS

def generate_recommendations(patient_df):
    
    recs = []

    if patient_df["number_inpatient"].iloc[0] >= 2:
        recs.append("Realizar seguimiento precoz tras el alta.")

    if patient_df["num_medications"].iloc[0] >= 15:
        recs.append("Revisar tratamiento farmacológico.")

    if patient_df["number_diagnoses"].iloc[0] >= 8:
        recs.append("Priorizar control de comorbilidades.")

    if patient_df["time_in_hospital"].iloc[0] >= 7:
        recs.append("Programar revisión clínica temprana.")

    if len(recs) == 0:
        recs.append("Mantener seguimiento habitual.")

    return recs

# PREDICCIÓN

if predict_button:

    # PACIENTE BASE
    patient = {}

    # Inicializamos todas las variables
    for feature in features:
        patient[feature] = 0
    
    # Variables categóricas
    for cat in cat_features:
        patient[cat] = "Unknown"

    # VARIABLES CATEGÓRICAS POR DEFECTO
    patient["weight"] = "?"
    patient["payer_code"] = "?"
    patient["medical_specialty"] = "InternalMedicine"
    patient["diag_1_group"] = "Circulatory"
    patient["diag_2_group"] = "Circulatory"
    patient["diag_3_group"] = "Circulatory"
    patient["change"] = "No"
    patient["diabetesMed"] = "Yes"
    patient["A1Cresult"] = a1c
    patient["max_glu_serum"] = max_glu

    medications = ["metformin", "repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton","insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone"]

    for med in medications:

        patient[med] = "No"
        patient["metformin"] = "Steady"
        patient["insulin"] = "Steady"

    patient["admission_type_id"] = 1
    patient["discharge_disposition_id"] = 1
    patient["admission_source_id"] = 1
    patient["num_procedures"] = 1
    
    # VARIABLES BÁSICAS

    patient["age"] = age
    patient["gender"] = gender
    patient["race"] = race
    patient["time_in_hospital"] = time_in_hospital
    patient["num_medications"] = num_medications
    patient["num_lab_procedures"] = num_lab_procedures
    patient["number_inpatient"] = number_inpatient
    patient["number_outpatient"] = number_outpatient
    patient["number_emergency"] = number_emergency
    patient["number_diagnoses"] = number_diagnoses

    # VARIABLES DERIVADAS

    patient["clinical_complexity"] = (num_medications+ number_diagnoses)
    patient["previous_utilization"] = (number_inpatient+ number_outpatient+ number_emergency)
    patient["hospital_intensity"] = (num_lab_procedures +patient["num_procedures"] +time_in_hospital)
    patient["medication_load"] = (num_medications* number_diagnoses)
    patient["medication_per_day"] = (num_medications /(time_in_hospital + 1))
    patient["lab_per_day"] = (num_lab_procedures /(time_in_hospital + 1))
    patient["diagnoses_per_day"] = (number_diagnoses /(time_in_hospital + 1))
    patient["polypharmacy"] = (1 if num_medications >= 10 else 0)
    patient["elderly"] = (1 if age in ["[70-80)", "[80-90)", "[90-100)"] else 0)
    patient["high_utilization"] = (1 if (number_inpatient +number_outpatient +number_emergency) >= 3 else 0)
    patient["multiple_admissions"] = (1 if number_inpatient >= 2 else 0)
    patient["high_risk_history"] = (1 if number_inpatient >= 3 else 0)
    patient["very_complex"] = (1 if number_diagnoses >= 8 else 0)

    # DATAFRAME
    patient_df = pd.DataFrame([patient])
    st.session_state["patient_df"] = patient_df

    # PREDICCIÓN
    risk = best_model.predict_proba(patient_df)[:,1][0]

    st.session_state["risk"] = risk

    # SHAP
    shap_values = explainer.shap_values(patient_df)
    explanation = pd.DataFrame({"variable": patient_df.columns,"shap_value": shap_values[0]})
    explanation["abs_shap"] = ( explanation["shap_value"].abs())
    explanation = explanation.sort_values("abs_shap",ascending=False)
    excluded_vars = ["weight","payer_code"]
    explanation = explanation[ ~explanation["variable"].isin(excluded_vars)]

    # DEBUG
    print("\nRIESGO:")
    print(risk)
    print("\nTOP 10 FACTORES SHAP")
    print(explanation[["variable","shap_value"]].head(10))

    # CLASIFICACIÓN
    if risk >= best_threshold:
        label = "ALTO RIESGO"

    else:
        label = "BAJO RIESGO"

    st.session_state["label"] = label

    # RESULTADOS
    st.divider()
    st.header("📊 Resultado")
    k1, k2, k3 = st.columns(3)
    k1.metric("Probabilidad",f"{risk*100:.2f}%")
    if label == "ALTO RIESGO":
        display_label = "🔴 ALTO RIESGO"
    else:
        display_label = "🟢 BAJO RIESGO"
    k2.metric("Clasificación", display_label)
    k3.metric( "Threshold", f"{best_threshold:.2f}")

    # VELOCÍMETRO
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            title={"text": "Riesgo de reingreso (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, best_threshold*100],"color": "lightgreen"},
                    {"range": [best_threshold*100, 70],"color": "khaki"},
                    {"range": [70, 100],"color": "lightcoral"}
                ]
            }
        )
    )

    st.plotly_chart(fig,use_container_width=True)

    # INTERPRETACIÓN
    st.subheader("🧠 Interpretación")

    if risk < 0.30:
        st.success("Riesgo bajo de reingreso hospitalario.")

    elif risk < best_threshold:
        st.warning("Riesgo clínico moderado.")
    
    else:
        st.error("Riesgo elevado de reingreso.")

    st.info(f"Probabilidad estimada: {risk*100:.2f}%")

    # FACTORES DE RIESGO
    st.divider()

    st.subheader("🔎 Factores más influyentes")

    top_factors = explanation.head(5)
    top_factors = top_factors[top_factors["abs_shap"] > 0.01]
    st.session_state["top_factors"] = top_factors

    for _, row in top_factors.iterrows():

        variable = variable_names.get(row["variable"],row["variable"])
        shap_value = row["shap_value"]

        if shap_value > 0:

            st.error(f"↑ {variable} "f"(incrementa el riesgo, SHAP = +{shap_value:.3f})")

        else:

            st.success(f"↓ {variable} "f"(reduce el riesgo, SHAP = {shap_value:.3f})")

    # GRÁFICO SHAP
    st.subheader("📈 Principales factores de riesgo")
    plot_df = top_factors.copy()
    plot_df["variable"] = plot_df["variable"].apply(lambda x: variable_names.get(x, x))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(plot_df["variable"],plot_df["shap_value"])
    ax.set_xlabel("Impacto SHAP")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

    # RECOMENDACIONES
    st.subheader("💡 Recomendaciones clínicas")
    recommendations = generate_recommendations(patient_df)
    st.session_state["recommendations"] = recommendations
    important_vars = top_factors["variable"].tolist()

    if "number_inpatient" in important_vars:
        recommendations.append("Reforzar seguimiento tras el alta por historial de hospitalizaciones.")

    if "number_emergency" in important_vars:
        recommendations.append("Valorar plan de prevención de visitas a urgencias.")

    if "medication_load" in important_vars:
        recommendations.append("Revisar posibles problemas derivados de polimedicación.")

    if "A1Cresult" in important_vars:
        recommendations.append("Mantener monitorización estrecha del control glucémico.")
    for rec in recommendations:
        st.success(rec)

    clinical_report = build_clinical_report(risk,label,recommendations,top_factors)
    st.session_state["clinical_report"] = clinical_report


with tab2:

    st.header( "📄 Informe clínico")

    if "clinical_report" in st.session_state:

        st.text_area("Informe generado",st.session_state["clinical_report"],height=400 )

    else:

        st.info("Realice primero una predicción para generar el informe.")

    if "clinical_report" in st.session_state:
        pdf_file = create_pdf_report(st.session_state["clinical_report"])
        st.download_button(label="📄 Descargar PDF",data=pdf_file,file_name="clinical_report.pdf",mime="application/pdf")


with tab3:

    st.header("🤖 AI Clinical Assistant")

    st.markdown( """Realice preguntas sobre el paciente, los factores de riesgo o la predicción obtenida por el modelo.""")

    if "assistant_history" not in st.session_state:

        st.session_state["assistant_history"] = []

    question = st.text_area("Pregunta", height=120)

    ask_button = st.button("Preguntar al asistente")


    if "risk" not in st.session_state:

        st.warning("Realice primero una predicción.")

    else:

        if ask_button:

            if question.strip() == "":

                st.warning("Introduzca una pregunta.")

            else:

                with st.spinner("Analizando..."):

                    answer = ask_ai_assistant(question,st.session_state["risk"],st.session_state["label"],st.session_state["top_factors"],st.session_state["recommendations"],st.session_state["patient_df"])

                st.session_state["assistant_history"].append(( question,answer))
    
        for q, a in reversed(st.session_state["assistant_history"]):

            st.markdown(f"### 👤 Usuario\n{q}")
            st.markdown( f"### 🤖 Asistente\n{a}")
            st.divider()

        
with tab4:

    st.header(
        "ℹ️ Sobre el modelo"
    )

    st.markdown(
        """
### Modelo predictivo

CatBoost Classifier optimizado mediante Optuna.

### Rendimiento obtenido

- AUC: 0.6708
- Accuracy: 0.7477
- Precision: 0.1993
- Recall: 0.4181
- F1 Score: 0.2699

### Variables más importantes

- Ingresos hospitalarios previos
- Utilización previa del sistema sanitario
- Historial clínico de riesgo
- Complejidad clínica
- HbA1c

### Objetivo

Identificar pacientes con mayor riesgo de
reingreso hospitalario para facilitar la
toma de decisiones clínicas.
"""
    )

    st.divider()

    st.caption("Esta herramienta tiene finalidad académica y de apoyo a la decisión clínica. No sustituye el criterio médico profesional.")