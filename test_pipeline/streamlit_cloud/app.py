"""
Preditor de Nível de Obesidade — Streamlit Cloud
Arquitetura A: modelo embutido via joblib (sem API separada)
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.joblib")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "Obesity.csv")
FEATURES     = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE",
                "Gender","family_history","FAVC","SMOKE","SCC","CAEC","CALC","MTRANS"]
CAEC_MAP     = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_MAP     = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MTRANS_MAP   = {"Automobile": 0, "Bike": 1, "Motorbike": 2,
                "Public_Transportation": 3, "Walking": 4}
COLOR_MAP    = {
    "Insufficient_Weight": ("🔵", "#2196F3"),
    "Normal_Weight":       ("🟢", "#4CAF50"),
    "Overweight_Level_I":  ("🟡", "#FFC107"),
    "Overweight_Level_II": ("🟠", "#FF9800"),
    "Obesity_Type_I":      ("🔴", "#F44336"),
    "Obesity_Type_II":     ("🔴", "#D32F2F"),
    "Obesity_Type_III":    ("🔴", "#B71C1C"),
}

# ── Treino (executado uma vez, resultado cacheado) ────────────────────────────
@st.cache_resource(show_spinner="Treinando modelo...")
def load_or_train_model():
    """Carrega o modelo se existir, senão treina e salva."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df_enc = df.copy()
    df_enc["Gender"]         = (df_enc["Gender"] == "Male").astype(int)
    df_enc["family_history"] = (df_enc["family_history"] == "yes").astype(int)
    df_enc["FAVC"]           = (df_enc["FAVC"] == "yes").astype(int)
    df_enc["SMOKE"]          = (df_enc["SMOKE"] == "yes").astype(int)
    df_enc["SCC"]            = (df_enc["SCC"] == "yes").astype(int)
    df_enc["CAEC"]           = df_enc["CAEC"].map(CAEC_MAP)
    df_enc["CALC"]           = df_enc["CALC"].map(CALC_MAP)
    df_enc["MTRANS"]         = df_enc["MTRANS"].map(MTRANS_MAP)

    X = df_enc[FEATURES]
    y = df_enc["Obesity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)

    return model, acc


def encode_inputs(gender, age, height, weight, family_history, favc,
                  fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans):
    return [
        age, height, weight, fcvc, ncp, ch2o, faf, tue,
        1 if gender == "Male" else 0,
        1 if family_history == "yes" else 0,
        1 if favc == "yes" else 0,
        1 if smoke == "yes" else 0,
        1 if scc == "yes" else 0,
        CAEC_MAP.get(caec, 0),
        CALC_MAP.get(calc, 0),
        MTRANS_MAP.get(mtrans, 3),
    ]


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Preditor de Obesidade",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃 Preditor de Nível de Obesidade")
st.markdown("Preencha os dados do paciente e clique em **Prever**.")

# Carrega/treina modelo
result = load_or_train_model()
if isinstance(result, tuple):
    model, acc = result
    st.caption(f"Modelo treinado — acurácia no teste: **{acc*100:.1f}%**")
else:
    model = result

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dados pessoais")
    gender         = st.selectbox("Gênero", ["Male", "Female"])
    age            = st.slider("Idade", 10, 80, 25)
    height         = st.slider("Altura (m)", 1.40, 2.00, 1.70, step=0.01)
    weight         = st.slider("Peso (kg)", 30.0, 180.0, 70.0, step=0.5)
    family_history = st.radio("Histórico familiar de obesidade?", ["yes", "no"],
                              horizontal=True)

with col2:
    st.subheader("Hábitos")
    favc   = st.radio("Consome alimentos calóricos frequentemente?",
                      ["yes", "no"], horizontal=True)
    fcvc   = st.slider("Frequência de vegetais nas refeições (1–3)", 1.0, 3.0, 2.0, step=0.5)
    ncp    = st.slider("Refeições principais por dia", 1.0, 4.0, 3.0, step=0.5)
    caec   = st.selectbox("Come entre refeições?",
                          ["no", "Sometimes", "Frequently", "Always"])
    smoke  = st.radio("Fuma?", ["no", "yes"], horizontal=True)
    ch2o   = st.slider("Litros de água por dia", 1.0, 3.0, 2.0, step=0.5)
    scc    = st.radio("Monitora calorias?", ["no", "yes"], horizontal=True)
    faf    = st.slider("Atividade física (dias/semana)", 0.0, 3.0, 1.0, step=0.5)
    tue    = st.slider("Horas de tecnologia por dia", 0.0, 2.0, 1.0, step=0.5)
    calc   = st.selectbox("Consome álcool?",
                          ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transporte principal",
                          ["Public_Transportation", "Walking",
                           "Automobile", "Bike", "Motorbike"])

st.divider()

if st.button("🔍 Prever nível de obesidade", use_container_width=True, type="primary"):
    features = encode_inputs(
        gender, age, height, weight, family_history, favc,
        fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans
    )

    prediction = model.predict([features])[0]
    proba      = model.predict_proba([features])[0]
    confidence = float(np.max(proba))

    icon, color = COLOR_MAP.get(prediction, ("⚪", "#9E9E9E"))
    label = prediction.replace("_", " ")

    st.markdown(
        f"""
        <div style="background:{color}22; border-left:6px solid {color};
                    padding:1rem 1.5rem; border-radius:8px; margin-top:1rem;">
            <h2 style="color:{color}; margin:0">{icon} {label}</h2>
            <p style="margin:0.5rem 0 0; color:#555">
                Confiança do modelo: <strong>{confidence*100:.1f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # BMI informativo
    bmi = weight / (height ** 2)
    st.metric("IMC calculado", f"{bmi:.1f} kg/m²")
