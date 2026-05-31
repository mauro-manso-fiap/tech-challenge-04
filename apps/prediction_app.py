"""
Aplicativo Streamlit de Predição de Obesidade.

Exibe formulário com 16 campos em PT-BR, chama PredictionService.predict()
e apresenta resultado estruturado com classificação, IMC, alertas contextuais,
riscos, recomendações e rodapé com metadados do modelo.
"""

import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Configuração de página — deve ser o primeiro comando Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="⚕️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Ajuste de path para importações relativas ao projeto
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (  # noqa: E402
    CAEC_OPTIONS,
    CALC_OPTIONS,
    GENDER_OPTIONS,
    MTRANS_OPTIONS,
    NUMERIC_RANGES,
    OBESITY_LABELS_PT,
    RESULT_INFO,
)
from src.services.prediction import PredictionService  # noqa: E402

# ---------------------------------------------------------------------------
# Cache do serviço de predição
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_service() -> PredictionService:
    """Carrega e cacheia o PredictionService com caminhos absolutos."""
    pipeline_path = os.path.join(PROJECT_ROOT, "models", "pipeline.joblib")
    metadata_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    return PredictionService(
        pipeline_path=pipeline_path,
        metadata_path=metadata_path,
    )


# ---------------------------------------------------------------------------
# Helpers de exibição
# ---------------------------------------------------------------------------

_BMI_THRESHOLDS = [
    (18.5, "Abaixo do Peso", "#3B82F6"),
    (24.9, "Normal", "#22C55E"),
    (29.9, "Sobrepeso", "#F97316"),
    (float("inf"), "Obesidade", "#EF4444"),
]

_RISK_COLORS = {
    "baixo": "#22C55E",
    "moderado": "#EAB308",
    "alto": "#EF4444",
    "muito_alto": "#B91C1C",
}

_RISK_LABELS = {
    "baixo": "Baixo",
    "moderado": "Moderado",
    "alto": "Alto",
    "muito_alto": "Muito Alto",
}


def _bmi_interpretation(bmi: float) -> tuple[str, str]:
    """Retorna (interpretação, cor_hex) para o IMC fornecido."""
    for threshold, label, color in _BMI_THRESHOLDS:
        if bmi <= threshold:
            return label, color
    return "Obesidade", "#B91C1C"


def _render_result_header(result) -> None:
    """Exibe cabeçalho colorido com ícone, classe e confiança."""
    info = RESULT_INFO[result.predicted_class]
    color = info["color"]
    icon = info["icon"]
    label = result.predicted_label_pt
    confidence_pct = result.confidence * 100
    risk_label = _RISK_LABELS.get(result.risk_level, result.risk_level)
    risk_color = _RISK_COLORS.get(result.risk_level, "#888888")

    st.markdown(
        f"""
        <div style="
            background-color: {color}22;
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <h2 style="margin: 0; color: {color};">{icon} {label}</h2>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; color: #444;">
                Confiança do modelo: <strong>{confidence_pct:.1f}%</strong>
                &nbsp;|&nbsp;
                Nível de risco:
                <span style="color: {risk_color}; font-weight: bold;">{risk_label}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_bmi_card(bmi: float) -> None:
    """Exibe card com valor do IMC e interpretação clínica."""
    interpretation, color = _bmi_interpretation(bmi)
    st.markdown(
        f"""
        <div style="
            background-color: {color}18;
            border: 1px solid {color}66;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <h4 style="margin: 0 0 4px 0; color: #333;">📊 Índice de Massa Corporal (IMC)</h4>
            <p style="margin: 0; font-size: 1.8rem; font-weight: bold; color: {color};">
                {bmi:.2f} kg/m²
            </p>
            <p style="margin: 4px 0 0 0; color: #555;">
                Classificação: <strong style="color: {color};">{interpretation}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_contextual_alerts(input_data: dict) -> None:
    """Exibe alertas contextuais baseados nas respostas do formulário."""
    family_history = input_data.get("family_history", "no")
    smoke = input_data.get("SMOKE", "no")
    faf = float(input_data.get("FAF", 1.0))
    favc = input_data.get("FAVC", "no")
    ch2o = float(input_data.get("CH2O", 2.0))

    if family_history == "yes":
        st.warning(
            "⚠️ **Histórico familiar de obesidade detectado.** "
            "Pessoas com histórico familiar têm risco significativamente maior de desenvolver obesidade. "
            "Acompanhamento preventivo regular é especialmente recomendado."
        )

    if smoke == "yes":
        st.warning(
            "🚬 **Tabagismo detectado.** "
            "O tabagismo aumenta o risco cardiovascular e metabólico, "
            "especialmente em combinação com excesso de peso. "
            "Considere programas de cessação do tabagismo."
        )

    if favc == "yes" and faf < 1.0:
        st.error(
            "⚠️ **Combinação de risco: alimentação calórica + sedentarismo.** "
            "O consumo frequente de alimentos calóricos aliado à baixa atividade física "
            "é um dos principais fatores de risco para obesidade e doenças metabólicas."
        )
    elif faf < 1.0:
        st.warning(
            "🏃 **Baixa atividade física detectada.** "
            "A OMS recomenda pelo menos 150 minutos de atividade física moderada por semana. "
            "Aumentar o nível de atividade física traz benefícios significativos para a saúde."
        )

    if ch2o < 1.5:
        st.warning(
            "💧 **Baixo consumo de água detectado.** "
            "A hidratação adequada (≥ 2 litros/dia) é essencial para o metabolismo saudável "
            "e pode auxiliar no controle do peso corporal."
        )


def _render_probability_chart(all_probabilities: dict) -> None:
    """Exibe gráfico de barras com probabilidades para todas as 7 classes."""
    import pandas as pd

    labels = [OBESITY_LABELS_PT.get(cls, cls) for cls in all_probabilities]
    values = [round(prob * 100, 2) for prob in all_probabilities.values()]

    df_chart = pd.DataFrame({"Classificação": labels, "Probabilidade (%)": values})
    df_chart = df_chart.sort_values("Probabilidade (%)", ascending=False)

    st.markdown("**📈 Distribuição de Probabilidades por Classe**")
    st.bar_chart(df_chart.set_index("Classificação"), height=280)


def _render_footer(result) -> None:
    """Exibe rodapé com metadados do modelo."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: #888; text-align: center; padding: 8px 0;">
            🤖 Modelo: <strong>{result.model_version}</strong>
            &nbsp;|&nbsp;
            📅 Data de treino: <strong>{result.training_date}</strong>
            &nbsp;|&nbsp;
            🎯 Acurácia no conjunto de teste: <strong>{result.accuracy * 100:.2f}%</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Ponto de entrada do aplicativo Streamlit."""

    # Título e descrição
    st.title("⚕️ Predição de Obesidade")
    st.markdown(
        "Preencha os dados abaixo para obter uma estimativa do nível de obesidade "
        "com base em um modelo de aprendizado de máquina treinado com dados clínicos e comportamentais."
    )

    # Disclaimer médico obrigatório
    st.warning(
        "⚕️ **Aviso:** Este aplicativo é uma ferramenta de apoio à decisão clínica "
        "baseada em aprendizado de máquina. Não substitui avaliação médica profissional. "
        "Os resultados devem ser interpretados por profissional de saúde habilitado."
    )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Formulário de entrada
    # -----------------------------------------------------------------------

    with st.form("prediction_form"):
        st.subheader("📋 Dados do Paciente")

        col1, col2, col3, col4 = st.columns(4)

        # --- Coluna 1 ---
        with col1:
            st.markdown("**Dados Pessoais**")

            gender_label = st.selectbox(
                "Gênero",
                options=list(GENDER_OPTIONS.keys()),
            )

            age = st.number_input(
                "Idade (anos)",
                min_value=int(NUMERIC_RANGES["Age"][0]),
                max_value=int(NUMERIC_RANGES["Age"][1]),
                value=30,
                step=1,
            )

            height = st.number_input(
                "Altura (m)",
                min_value=float(NUMERIC_RANGES["Height"][0]),
                max_value=float(NUMERIC_RANGES["Height"][1]),
                value=1.70,
                step=0.01,
                format="%.2f",
            )

            weight = st.number_input(
                "Peso (kg)",
                min_value=float(NUMERIC_RANGES["Weight"][0]),
                max_value=float(NUMERIC_RANGES["Weight"][1]),
                value=70.0,
                step=0.5,
                format="%.1f",
            )

        # --- Coluna 2 ---
        with col2:
            st.markdown("**Hábitos Alimentares**")

            family_history_label = st.selectbox(
                "Histórico Familiar de Obesidade",
                options=["Sim", "Não"],
            )

            favc_label = st.selectbox(
                "Consumo Frequente de Alimentos Calóricos",
                options=["Sim", "Não"],
            )

            fcvc = st.number_input(
                "Freq. Consumo de Vegetais (0–3)",
                min_value=float(NUMERIC_RANGES["FCVC"][0]),
                max_value=float(NUMERIC_RANGES["FCVC"][1]),
                value=2.0,
                step=0.5,
                format="%.1f",
                help="0 = Nunca, 1 = Às vezes, 2 = Frequentemente, 3 = Sempre",
            )

            ncp = st.number_input(
                "Nº de Refeições Principais por Dia",
                min_value=float(NUMERIC_RANGES["NCP"][0]),
                max_value=float(NUMERIC_RANGES["NCP"][1]),
                value=3.0,
                step=0.5,
                format="%.1f",
                help="Número de refeições principais realizadas por dia",
            )

        # --- Coluna 3 ---
        with col3:
            st.markdown("**Comportamento e Saúde**")

            caec_label = st.selectbox(
                "Consumo de Alimentos entre Refeições",
                options=list(CAEC_OPTIONS.keys()),
            )

            smoke_label = st.selectbox(
                "Tabagismo",
                options=["Não", "Sim"],
            )

            ch2o = st.number_input(
                "Consumo Diário de Água (0–3)",
                min_value=float(NUMERIC_RANGES["CH2O"][0]),
                max_value=float(NUMERIC_RANGES["CH2O"][1]),
                value=2.0,
                step=0.5,
                format="%.1f",
                help="0 = Menos de 1L, 1 = Entre 1–2L, 2 = Entre 2–3L, 3 = Mais de 3L",
            )

            scc_label = st.selectbox(
                "Monitoramento de Calorias Consumidas",
                options=["Não", "Sim"],
            )

        # --- Coluna 4 ---
        with col4:
            st.markdown("**Atividade Física e Transporte**")

            faf = st.number_input(
                "Freq. de Atividade Física (0–3)",
                min_value=float(NUMERIC_RANGES["FAF"][0]),
                max_value=float(NUMERIC_RANGES["FAF"][1]),
                value=1.0,
                step=0.5,
                format="%.1f",
                help="0 = Nenhuma, 1 = 1–2 dias/semana, 2 = 3–4 dias/semana, 3 = 4–5 dias/semana",
            )

            tue = st.number_input(
                "Tempo de Uso de Tecnologia/dia (0–2)",
                min_value=float(NUMERIC_RANGES["TUE"][0]),
                max_value=float(NUMERIC_RANGES["TUE"][1]),
                value=1.0,
                step=0.5,
                format="%.1f",
                help="0 = 0–2h, 1 = 3–5h, 2 = Mais de 5h",
            )

            calc_label = st.selectbox(
                "Consumo de Álcool",
                options=list(CALC_OPTIONS.keys()),
            )

            mtrans_label = st.selectbox(
                "Principal Meio de Transporte",
                options=list(MTRANS_OPTIONS.keys()),
            )

        st.markdown("")
        submitted = st.form_submit_button(
            "🔍 Analisar Perfil",
            use_container_width=True,
            type="primary",
        )

    # -----------------------------------------------------------------------
    # Processamento e exibição do resultado
    # -----------------------------------------------------------------------

    if submitted:
        # Converter labels PT-BR para valores do dataset
        input_data = {
            "Gender": GENDER_OPTIONS[gender_label],
            "Age": float(age),
            "Height": float(height),
            "Weight": float(weight),
            "family_history": "yes" if family_history_label == "Sim" else "no",
            "FAVC": "yes" if favc_label == "Sim" else "no",
            "FCVC": float(fcvc),
            "NCP": float(ncp),
            "CAEC": CAEC_OPTIONS[caec_label],
            "SMOKE": "yes" if smoke_label == "Sim" else "no",
            "CH2O": float(ch2o),
            "SCC": "yes" if scc_label == "Sim" else "no",
            "FAF": float(faf),
            "TUE": float(tue),
            "CALC": CALC_OPTIONS[calc_label],
            "MTRANS": MTRANS_OPTIONS[mtrans_label],
        }

        try:
            service = _load_service()
            result = service.predict(input_data)
        except FileNotFoundError as exc:
            st.error(
                f"❌ **Modelo não encontrado.** "
                f"Execute o notebook de treinamento para gerar o arquivo `pipeline.joblib`.\n\n"
                f"Detalhe: {exc}"
            )
            return
        except ValueError as exc:
            st.error(f"❌ **Dados inválidos:** {exc}")
            return

        st.markdown("---")
        st.subheader("📊 Resultado da Análise")

        # a) Cabeçalho com classificação, ícone, cor e confiança
        _render_result_header(result)

        col_bmi, col_desc = st.columns([1, 2])

        with col_bmi:
            # b) Card de IMC
            _render_bmi_card(result.bmi)

        with col_desc:
            # c) Descrição clínica
            info = RESULT_INFO[result.predicted_class]
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 16px 20px;
                    margin-bottom: 16px;
                    height: 100%;
                ">
                    <h4 style="margin: 0 0 8px 0; color: #333;">🩺 Descrição Clínica</h4>
                    <p style="margin: 0; color: #555; line-height: 1.6;">{info['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # d) Alertas contextuais
        _render_contextual_alerts(input_data)

        # e) Riscos e f) Recomendações lado a lado
        col_risks, col_recs = st.columns(2)

        with col_risks:
            st.markdown("**⚠️ Riscos Associados**")
            for risk in info["risks"]:
                st.markdown(f"- {risk}")

        with col_recs:
            st.markdown("**✅ Recomendações**")
            for rec in info["recommendations"]:
                st.markdown(f"- {rec}")

        # g) Gráfico de probabilidades
        st.markdown("")
        _render_probability_chart(result.all_probabilities)

        # h) Rodapé com metadados do modelo
        _render_footer(result)


if __name__ == "__main__":
    main()
