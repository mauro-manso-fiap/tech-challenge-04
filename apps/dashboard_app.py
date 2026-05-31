"""
Painel Analítico — Análise de Obesidade
========================================
App 2 do Tech Challenge 04 — PosTech Data Analytics (FIAP).

Exibe análises exploratórias interativas sobre o dataset de obesidade,
com filtros por gênero, faixa etária e classe de obesidade.

Seções:
    1. Visão Geral — distribuição das 7 classes
    2. Correlações de Spearman — features × target
    3. Análise Demográfica — idade e gênero por classe
    4. Impacto do Histórico Familiar — risco por histórico
    5. Fatores de Risco — top 5 fatores associados à obesidade
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Caminho absoluto para o projeto (garante funcionamento em qualquer CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (  # noqa: E402
    CATEGORICAL_VALUES,
    GENDER_OPTIONS,
    NUMERIC_RANGES,
    OBESITY_LABELS_PT,
    RESULT_INFO,
)
from src.services.analytics import AnalyticsService  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Painel Analítico — Obesidade",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inicialização do serviço (cacheado para não recarregar a cada interação)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_service() -> AnalyticsService:
    data_path = os.path.join(PROJECT_ROOT, "data", "Obesity.csv")
    return AnalyticsService(data_path=data_path)


service = _load_service()

# ---------------------------------------------------------------------------
# Helpers de dados cacheados
# ---------------------------------------------------------------------------

@st.cache_data
def _get_class_distribution() -> pd.DataFrame:
    return service.get_class_distribution()


@st.cache_data
def _get_spearman_correlations() -> pd.DataFrame:
    return service.get_spearman_correlations()


@st.cache_data
def _get_demographic_breakdown() -> dict:
    return service.get_demographic_breakdown()


@st.cache_data
def _get_family_history_impact() -> pd.DataFrame:
    return service.get_family_history_impact()


@st.cache_data
def _get_top_risk_factors(n: int = 5) -> pd.DataFrame:
    return service.get_top_risk_factors(n=n)


# ---------------------------------------------------------------------------
# Cabeçalho principal
# ---------------------------------------------------------------------------
st.title("📊 Painel Analítico — Análise de Obesidade")
st.markdown(
    """
    Este painel apresenta análises exploratórias interativas sobre o dataset de obesidade,
    permitindo identificar padrões populacionais, fatores de risco e distribuições demográficas.
    Use os filtros na barra lateral para segmentar os dados conforme necessário.
    """
)

# ---------------------------------------------------------------------------
# Barra lateral — Filtros interativos
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filtros")

# Gênero
genero_opcoes_pt = list(GENDER_OPTIONS.keys())  # ["Masculino", "Feminino"]
genero_selecionado_pt = st.sidebar.multiselect(
    "Gênero",
    options=genero_opcoes_pt,
    default=genero_opcoes_pt,
    help="Selecione um ou mais gêneros para filtrar os dados.",
)
genero_en = [GENDER_OPTIONS[g] for g in genero_selecionado_pt]

# Faixa etária
age_min_global = int(NUMERIC_RANGES["Age"][0])
age_max_global = int(NUMERIC_RANGES["Age"][1])

st.sidebar.markdown("**Faixa Etária**")
col_age1, col_age2 = st.sidebar.columns(2)
with col_age1:
    idade_min = st.number_input(
        "Mínima",
        min_value=age_min_global,
        max_value=age_max_global,
        value=age_min_global,
        step=1,
        key="idade_min",
    )
with col_age2:
    idade_max = st.number_input(
        "Máxima",
        min_value=age_min_global,
        max_value=age_max_global,
        value=age_max_global,
        step=1,
        key="idade_max",
    )

# Classes de obesidade
classes_pt = list(OBESITY_LABELS_PT.values())
classes_en = list(OBESITY_LABELS_PT.keys())
classes_selecionadas_pt = st.sidebar.multiselect(
    "Classes de Obesidade",
    options=classes_pt,
    default=classes_pt,
    help="Selecione as classes de obesidade a incluir na análise.",
)
# Mapear PT → EN para o filtro
_pt_to_en = {v: k for k, v in OBESITY_LABELS_PT.items()}
classes_selecionadas_en = [_pt_to_en[c] for c in classes_selecionadas_pt]

# Botão de aplicar filtros
aplicar = st.sidebar.button("✅ Aplicar Filtros", use_container_width=True)

# Montar dicionário de filtros
_filters: dict = {}
if genero_en:
    _filters["gender"] = genero_en
if idade_min is not None:
    _filters["age_min"] = float(idade_min)
if idade_max is not None:
    _filters["age_max"] = float(idade_max)
if classes_selecionadas_en:
    _filters["obesity_classes"] = classes_selecionadas_en

# Dados filtrados (recalculados quando filtros mudam)
@st.cache_data
def _get_filtered(filters_key: str, filters: dict) -> pd.DataFrame:  # noqa: ARG001
    return service.get_filtered_data(filters)


# Usar uma chave estável baseada nos filtros para o cache
_filters_key = str(sorted(_filters.items()))
df_filtered = _get_filtered(_filters_key, _filters)

# Aviso se nenhum dado após filtros
if df_filtered.empty:
    st.warning("⚠️ Nenhum registro encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# Abas principais
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Visão Geral",
    "🔗 Correlações de Spearman",
    "👥 Análise Demográfica",
    "🧬 Histórico Familiar",
    "⚠️ Fatores de Risco",
])

# ===========================================================================
# ABA 1 — VISÃO GERAL
# ===========================================================================
with tab1:
    st.header("Visão Geral — Distribuição das Classes de Obesidade")

    dist_df = _get_class_distribution()

    # Filtrar distribuição pelas classes selecionadas
    dist_filtered = dist_df[dist_df["class_en"].isin(classes_selecionadas_en)].copy()

    # --- KPIs ---
    total_registros = len(df_filtered)
    n_classes = dist_filtered[dist_filtered["count"] > 0]["class_en"].nunique()
    if not dist_filtered.empty and dist_filtered["count"].sum() > 0:
        classe_mais_freq_row = dist_filtered.loc[dist_filtered["count"].idxmax()]
        pct_mais_freq = round(
            dist_filtered["count"].max() / dist_filtered["count"].sum() * 100, 1
        )
        classe_mais_freq_label = classe_mais_freq_row["class_pt"]
    else:
        pct_mais_freq = 0.0
        classe_mais_freq_label = "—"

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total de Registros", f"{total_registros:,}".replace(",", "."))
    kpi2.metric("Nº de Classes", n_classes)
    kpi3.metric(
        "Classe Mais Frequente",
        classe_mais_freq_label,
        delta=f"{pct_mais_freq}% dos registros",
        delta_color="off",
    )

    st.divider()

    # Paleta de cores por classe (usando RESULT_INFO)
    _color_map = {
        info["label_pt"]: info["color"]
        for key, info in RESULT_INFO.items()
    }

    col_bar, col_pie = st.columns(2)

    with col_bar:
        st.subheader("Distribuição por Classe (Contagem)")
        fig_bar = px.bar(
            dist_filtered,
            x="class_pt",
            y="count",
            color="class_pt",
            color_discrete_map=_color_map,
            labels={"class_pt": "Classe de Obesidade", "count": "Quantidade de Registros"},
            text="count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            xaxis_tickangle=-30,
            margin=dict(t=20, b=10),
            xaxis_title="Classe de Obesidade",
            yaxis_title="Quantidade",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        st.subheader("Proporção das Classes (%)")
        fig_pie = px.pie(
            dist_filtered,
            names="class_pt",
            values="count",
            color="class_pt",
            color_discrete_map=_color_map,
            hole=0.35,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="v", x=1.0, y=0.5),
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.caption(
        "Fonte: Dataset Obesity (2.111 registros originais). "
        "Os valores exibidos refletem a distribuição global do dataset, "
        "independentemente dos filtros de faixa etária e gênero aplicados."
    )

# ===========================================================================
# ABA 2 — CORRELAÇÕES DE SPEARMAN
# ===========================================================================
with tab2:
    st.header("Correlações de Spearman — Features × Nível de Obesidade")
    st.markdown(
        """
        O coeficiente de Spearman mede a associação monotônica entre cada variável e o nível de obesidade
        (codificado ordinalmente de 0 = Abaixo do Peso a 6 = Obesidade Tipo III).

        - **Correlação positiva** (barras em vermelho/laranja): valores mais altos da feature estão associados
          a níveis mais elevados de obesidade.
        - **Correlação negativa** (barras em azul): valores mais altos da feature estão associados
          a níveis mais baixos de obesidade.
        """
    )

    corr_df = _get_spearman_correlations()

    # Ordenar por correlação (não por valor absoluto) para visualização horizontal
    corr_sorted = corr_df.sort_values("correlation", ascending=True).copy()

    # Cor baseada no sinal da correlação
    corr_sorted["cor"] = corr_sorted["correlation"].apply(
        lambda v: "#EF4444" if v >= 0 else "#3B82F6"
    )

    fig_corr = go.Figure(
        go.Bar(
            x=corr_sorted["correlation"],
            y=corr_sorted["feature"],
            orientation="h",
            marker_color=corr_sorted["cor"].tolist(),
            text=corr_sorted["correlation"].apply(lambda v: f"{v:+.3f}"),
            textposition="outside",
        )
    )
    fig_corr.update_layout(
        xaxis_title="Correlação de Spearman",
        yaxis_title="Feature",
        xaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinewidth=2, zerolinecolor="gray"),
        margin=dict(t=20, b=10, l=120),
        height=500,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
        "💡 **Interpretação:** Peso corporal (Weight) e IMC (calculado internamente) apresentam as "
        "correlações positivas mais fortes com o nível de obesidade. Frequência de atividade física (FAF) "
        "apresenta correlação negativa, indicando que maior atividade está associada a menor risco."
    )

# ===========================================================================
# ABA 3 — ANÁLISE DEMOGRÁFICA
# ===========================================================================
with tab3:
    st.header("Análise Demográfica — Idade e Gênero por Classe de Obesidade")
    st.markdown(
        """
        Esta seção explora como a distribuição de **idade** e **gênero** varia entre as diferentes
        classes de obesidade, permitindo identificar perfis demográficos de risco.
        """
    )

    demo = _get_demographic_breakdown()
    age_df = demo["age_by_class"]
    gender_df = demo["gender_by_class"]

    # Filtrar pelas classes selecionadas
    classes_pt_selecionadas = [OBESITY_LABELS_PT[c] for c in classes_selecionadas_en]
    age_df_f = age_df[age_df["class_pt"].isin(classes_pt_selecionadas)].copy()
    gender_df_f = gender_df[gender_df["class_pt"].isin(classes_pt_selecionadas)].copy()

    col_age, col_gender = st.columns(2)

    with col_age:
        st.subheader("Média de Idade por Classe")
        fig_age = px.bar(
            age_df_f,
            x="class_pt",
            y="mean_age",
            color="class_pt",
            color_discrete_map=_color_map,
            labels={"class_pt": "Classe de Obesidade", "mean_age": "Média de Idade (anos)"},
            text=age_df_f["mean_age"].apply(lambda v: f"{v:.1f}"),
        )
        fig_age.update_traces(textposition="outside")
        fig_age.update_layout(
            showlegend=False,
            xaxis_tickangle=-30,
            margin=dict(t=20, b=10),
            yaxis_title="Média de Idade (anos)",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_gender:
        st.subheader("Distribuição de Gênero por Classe (%)")
        # Transformar para formato longo para plotly
        gender_long = gender_df_f.melt(
            id_vars="class_pt",
            value_vars=["Male_pct", "Female_pct"],
            var_name="Gênero",
            value_name="Percentual",
        )
        gender_long["Gênero"] = gender_long["Gênero"].map(
            {"Male_pct": "Masculino", "Female_pct": "Feminino"}
        )
        fig_gender = px.bar(
            gender_long,
            x="class_pt",
            y="Percentual",
            color="Gênero",
            barmode="group",
            color_discrete_map={"Masculino": "#3B82F6", "Feminino": "#EC4899"},
            labels={"class_pt": "Classe de Obesidade", "Percentual": "Percentual (%)"},
            text=gender_long["Percentual"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_gender.update_traces(textposition="outside")
        fig_gender.update_layout(
            xaxis_tickangle=-30,
            margin=dict(t=20, b=10),
            yaxis_title="Percentual (%)",
            legend_title="Gênero",
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    st.info(
        "💡 **Interpretação:** Classes de obesidade mais graves tendem a apresentar médias de idade "
        "ligeiramente superiores, sugerindo que o risco acumulado aumenta com a idade. "
        "A distribuição de gênero varia entre as classes, com diferenças notáveis nos extremos do espectro."
    )

# ===========================================================================
# ABA 4 — IMPACTO DO HISTÓRICO FAMILIAR
# ===========================================================================
with tab4:
    st.header("Impacto do Histórico Familiar no Risco de Obesidade")
    st.markdown(
        """
        O histórico familiar de obesidade é um dos fatores de risco mais relevantes.
        O gráfico abaixo compara a distribuição das classes de obesidade entre pessoas
        **com** e **sem** histórico familiar, revelando o impacto desse fator genético/ambiental.
        """
    )

    fh_df = _get_family_history_impact()
    fh_filtered = fh_df[fh_df["class_pt"].isin(classes_pt_selecionadas)].copy()

    # Formato longo para plotly
    fh_long = fh_filtered.melt(
        id_vars="class_pt",
        value_vars=["with_history_pct", "without_history_pct"],
        var_name="Histórico Familiar",
        value_name="Percentual (%)",
    )
    fh_long["Histórico Familiar"] = fh_long["Histórico Familiar"].map(
        {"with_history_pct": "Com histórico familiar", "without_history_pct": "Sem histórico familiar"}
    )

    fig_fh = px.bar(
        fh_long,
        x="class_pt",
        y="Percentual (%)",
        color="Histórico Familiar",
        barmode="group",
        color_discrete_map={
            "Com histórico familiar": "#EF4444",
            "Sem histórico familiar": "#22C55E",
        },
        labels={"class_pt": "Classe de Obesidade"},
        text=fh_long["Percentual (%)"].apply(lambda v: f"{v:.1f}%"),
    )
    fig_fh.update_traces(textposition="outside")
    fig_fh.update_layout(
        xaxis_tickangle=-30,
        margin=dict(t=20, b=10),
        yaxis_title="Percentual dentro do grupo (%)",
        legend_title="Histórico Familiar",
        height=480,
    )
    st.plotly_chart(fig_fh, use_container_width=True)

    # Insight principal
    with_obesity_pct = fh_df[
        fh_df["class_pt"].isin(["Obesidade Tipo I", "Obesidade Tipo II", "Obesidade Tipo III"])
    ]["with_history_pct"].sum()
    without_obesity_pct = fh_df[
        fh_df["class_pt"].isin(["Obesidade Tipo I", "Obesidade Tipo II", "Obesidade Tipo III"])
    ]["without_history_pct"].sum()

    st.success(
        f"🧬 **Insight principal:** Pessoas **com** histórico familiar concentram "
        f"**{with_obesity_pct:.1f}%** dos casos nas três classes de obesidade (Tipos I, II e III), "
        f"contra **{without_obesity_pct:.1f}%** entre aquelas **sem** histórico familiar. "
        "Isso evidencia que o histórico familiar é um preditor significativo de obesidade grave."
    )

# ===========================================================================
# ABA 5 — FATORES DE RISCO
# ===========================================================================
with tab5:
    st.header("Top 5 Fatores de Risco Associados à Obesidade")
    st.markdown(
        """
        Os fatores abaixo foram identificados como os **mais fortemente associados** ao nível de obesidade,
        com base na correlação de Spearman entre cada variável e o target (ordenado de Abaixo do Peso
        a Obesidade Tipo III).
        """
    )

    risk_df = _get_top_risk_factors(n=5)

    # Gráfico de barras horizontal com cor por direção
    risk_df["cor"] = risk_df["correlation"].apply(
        lambda v: "#EF4444" if v >= 0 else "#3B82F6"
    )
    risk_df["direcao_label"] = risk_df["direction"].map(
        {"positivo": "Correlação Positiva", "negativo": "Correlação Negativa"}
    )

    fig_risk = go.Figure(
        go.Bar(
            x=risk_df["correlation"],
            y=risk_df["feature"],
            orientation="h",
            marker_color=risk_df["cor"].tolist(),
            text=risk_df["correlation"].apply(lambda v: f"{v:+.3f}"),
            textposition="outside",
        )
    )
    fig_risk.update_layout(
        xaxis_title="Correlação de Spearman",
        yaxis_title="Feature",
        xaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinewidth=2, zerolinecolor="gray"),
        margin=dict(t=20, b=10, l=120),
        height=350,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Tabela detalhada com interpretações
    st.subheader("Interpretação dos Fatores de Risco")

    for _, row in risk_df.iterrows():
        cor_badge = "🔴" if row["direction"] == "positivo" else "🔵"
        direcao_texto = "positiva" if row["direction"] == "positivo" else "negativa"
        with st.container():
            st.markdown(
                f"{cor_badge} **{row['feature']}** "
                f"(correlação {direcao_texto}: `{row['correlation']:+.3f}`)"
            )
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{row['interpretation_pt']}")
            st.divider()

    st.info(
        "💡 **Como interpretar:** Fatores com correlação **positiva** (vermelho) indicam que valores "
        "mais altos dessa variável estão associados a maior risco de obesidade. "
        "Fatores com correlação **negativa** (azul) indicam que valores mais altos estão associados "
        "a menor risco — como a frequência de atividade física (FAF)."
    )

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "📊 Painel Analítico — Obesidade | Tech Challenge 04 — PosTech Data Analytics (FIAP) | "
    "Dados: Obesity Dataset (2.111 registros × 17 variáveis)"
)
