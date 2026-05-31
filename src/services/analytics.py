"""
AnalyticsService — serviço de analytics para o painel analítico de obesidade.

Carrega o dataset Obesity.csv e expõe métodos para análise exploratória
usados pelo dashboard Streamlit (apps/dashboard_app.py).

Uso:
    from src.services.analytics import AnalyticsService

    svc = AnalyticsService()
    dist = svc.get_class_distribution()
    corr = svc.get_spearman_correlations()
"""

import os
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats

from src.config import (
    CATEGORICAL_VALUES,
    NUMERIC_RANGES,
    OBESITY_LABELS_PT,
)
from src.logging_config import setup_logging

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Constantes internas de encoding (sem hard-code fora do módulo)
# ---------------------------------------------------------------------------

# Ordem canônica das classes de obesidade (do mais leve ao mais grave)
_OBESITY_ORDER: List[str] = list(OBESITY_LABELS_PT.keys())

# Encoding ordinal do target para cálculo de correlação de Spearman
_OBESITY_ORDINAL: Dict[str, int] = {cls: i for i, cls in enumerate(_OBESITY_ORDER)}

# Encoding ordinal de CAEC e CALC
_FREQ_ORDINAL: Dict[str, int] = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

# Colunas binárias (yes/no ou Male/Female) codificadas como 0/1
_BINARY_COLS: Dict[str, Dict[str, int]] = {
    "Gender": {"Male": 1, "Female": 0},
    "family_history": {"yes": 1, "no": 0},
    "FAVC": {"yes": 1, "no": 0},
    "SMOKE": {"yes": 1, "no": 0},
    "SCC": {"yes": 1, "no": 0},
}

# Interpretações PT-BR para os fatores de risco mais comuns
_FEATURE_INTERPRETATIONS_PT: Dict[str, str] = {
    "Weight": "Peso corporal elevado está diretamente associado à obesidade",
    "BMI_proxy": "IMC elevado é o principal indicador de obesidade",
    "Height": "Altura influencia o IMC e a classificação de peso",
    "Age": "Idade avançada está associada a maior risco de sobrepeso",
    "FCVC": "Baixo consumo de vegetais está associado a pior qualidade alimentar",
    "NCP": "Número de refeições principais influencia o balanço calórico",
    "CH2O": "Baixa ingestão de água está associada a hábitos alimentares menos saudáveis",
    "FAF": "Baixa frequência de atividade física é fator de risco independente",
    "TUE": "Tempo excessivo em telas está associado a sedentarismo",
    "family_history": "Histórico familiar de obesidade aumenta significativamente o risco",
    "FAVC": "Consumo frequente de alimentos calóricos está associado à obesidade",
    "SMOKE": "Tabagismo está associado a alterações metabólicas e de peso",
    "SCC": "Monitoramento calórico está associado a maior consciência alimentar",
    "Gender": "Gênero influencia a distribuição de gordura corporal",
    "CAEC": "Consumo de alimentos entre refeições contribui para excesso calórico",
    "CALC": "Consumo de álcool adiciona calorias e afeta o metabolismo",
}


class AnalyticsService:
    """Serviço de analytics para o painel analítico de obesidade.

    Carrega e limpa o dataset na inicialização e expõe métodos de análise
    para uso no dashboard Streamlit.

    Args:
        data_path: Caminho para o arquivo CSV. Padrão: ``data/Obesity.csv``
                   relativo à raiz do projeto.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        if data_path is None:
            # Caminho relativo à raiz do projeto (dois níveis acima deste arquivo)
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            data_path = os.path.join(project_root, "data", "Obesity.csv")

        logger.info("Carregando dataset: %s", data_path)
        df = pd.read_csv(data_path)

        # Limpeza básica: remover duplicatas
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info("Duplicatas removidas: %d", n_removed)

        # Garantir que a coluna Obesity existe e tem apenas valores conhecidos
        df = df[df["Obesity"].isin(_OBESITY_ORDER)].copy()

        self.df: pd.DataFrame = df
        logger.info(
            "Dataset carregado com sucesso: %d registros, %d colunas",
            len(self.df),
            len(self.df.columns),
        )

    # ------------------------------------------------------------------
    # Distribuição de classes
    # ------------------------------------------------------------------

    def get_class_distribution(self) -> pd.DataFrame:
        """Retorna a distribuição das classes de obesidade.

        Returns:
            DataFrame com colunas: ``class_en``, ``class_pt``, ``count``,
            ``percentage``. Ordenado pela ordem canônica de OBESITY_LABELS_PT
            (do mais leve ao mais grave).
        """
        counts = self.df["Obesity"].value_counts()
        total = len(self.df)

        rows = []
        for class_en in _OBESITY_ORDER:
            count = int(counts.get(class_en, 0))
            rows.append(
                {
                    "class_en": class_en,
                    "class_pt": OBESITY_LABELS_PT[class_en],
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Correlações de Spearman
    # ------------------------------------------------------------------

    def get_spearman_correlations(self) -> pd.DataFrame:
        """Calcula a correlação de Spearman entre features e o target Obesity.

        O target é codificado ordinalmente (Insufficient_Weight=0 … Obesity_Type_III=6).
        Binárias (Gender, family_history, FAVC, SMOKE, SCC) são codificadas como 0/1.
        CAEC e CALC são codificadas ordinalmente (no=0, Sometimes=1, Frequently=2, Always=3).

        Returns:
            DataFrame com colunas: ``feature``, ``correlation``, ``abs_correlation``.
            Ordenado por ``abs_correlation`` decrescente.
        """
        df_enc = self._encode_for_correlation()
        target = df_enc["Obesity_encoded"]

        rows = []
        feature_cols = [c for c in df_enc.columns if c != "Obesity_encoded"]
        for col in feature_cols:
            corr_val, _ = stats.spearmanr(df_enc[col], target)
            rows.append(
                {
                    "feature": col,
                    "correlation": round(float(corr_val), 4),
                    "abs_correlation": round(abs(float(corr_val)), 4),
                }
            )

        result = pd.DataFrame(rows)
        result = result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Breakdown demográfico
    # ------------------------------------------------------------------

    def get_demographic_breakdown(self) -> Dict[str, pd.DataFrame]:
        """Retorna breakdown demográfico por classe de obesidade.

        Returns:
            Dicionário com:

            - ``'age_by_class'``: DataFrame com média e mediana de idade por classe
              (colunas: ``class_pt``, ``mean_age``, ``median_age``).
            - ``'gender_by_class'``: DataFrame com distribuição de gênero por classe
              em percentual (colunas: ``class_pt``, ``Male_pct``, ``Female_pct``).
        """
        df = self.df.copy()
        df["class_pt"] = df["Obesity"].map(OBESITY_LABELS_PT)

        # Idade por classe
        age_stats = (
            df.groupby("class_pt", observed=True)["Age"]
            .agg(mean_age="mean", median_age="median")
            .round(1)
            .reset_index()
        )
        # Reordenar pela ordem canônica
        class_pt_order = [OBESITY_LABELS_PT[c] for c in _OBESITY_ORDER]
        age_stats["class_pt"] = pd.Categorical(
            age_stats["class_pt"], categories=class_pt_order, ordered=True
        )
        age_stats = age_stats.sort_values("class_pt").reset_index(drop=True)

        # Gênero por classe (%)
        gender_counts = (
            df.groupby(["class_pt", "Gender"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        gender_pct = gender_counts.div(gender_counts.sum(axis=1), axis=0).mul(100).round(1)
        # Garantir que ambas as colunas existem
        for col in CATEGORICAL_VALUES["Gender"]:
            if col not in gender_pct.columns:
                gender_pct[col] = 0.0
        gender_pct = gender_pct.rename(
            columns={"Male": "Male_pct", "Female": "Female_pct"}
        ).reset_index()
        gender_pct["class_pt"] = pd.Categorical(
            gender_pct["class_pt"], categories=class_pt_order, ordered=True
        )
        gender_pct = gender_pct.sort_values("class_pt").reset_index(drop=True)

        return {
            "age_by_class": age_stats,
            "gender_by_class": gender_pct,
        }

    # ------------------------------------------------------------------
    # Impacto do histórico familiar
    # ------------------------------------------------------------------

    def get_family_history_impact(self) -> pd.DataFrame:
        """Retorna a distribuição de classes de obesidade por histórico familiar.

        Returns:
            DataFrame com colunas: ``class_pt``, ``with_history_pct``,
            ``without_history_pct``. Ordenado pela ordem canônica de classes.
        """
        df = self.df.copy()
        df["class_pt"] = df["Obesity"].map(OBESITY_LABELS_PT)

        with_hist = df[df["family_history"] == "yes"]["class_pt"].value_counts(normalize=True) * 100
        without_hist = df[df["family_history"] == "no"]["class_pt"].value_counts(normalize=True) * 100

        class_pt_order = [OBESITY_LABELS_PT[c] for c in _OBESITY_ORDER]
        rows = []
        for class_pt in class_pt_order:
            rows.append(
                {
                    "class_pt": class_pt,
                    "with_history_pct": round(float(with_hist.get(class_pt, 0.0)), 2),
                    "without_history_pct": round(float(without_hist.get(class_pt, 0.0)), 2),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Top fatores de risco
    # ------------------------------------------------------------------

    def get_top_risk_factors(self, n: int = 5) -> pd.DataFrame:
        """Retorna os top-n fatores mais associados à obesidade por correlação de Spearman.

        Args:
            n: Número de fatores a retornar. Padrão: 5.

        Returns:
            DataFrame com colunas: ``feature``, ``correlation``, ``direction``,
            ``interpretation_pt``. Ordenado por ``abs_correlation`` decrescente.
        """
        corr_df = self.get_spearman_correlations().head(n)

        rows = []
        for _, row in corr_df.iterrows():
            feature = row["feature"]
            corr_val = row["correlation"]
            direction = "positivo" if corr_val >= 0 else "negativo"
            interpretation = _FEATURE_INTERPRETATIONS_PT.get(
                feature,
                f"Feature '{feature}' apresenta correlação {direction} com obesidade",
            )
            rows.append(
                {
                    "feature": feature,
                    "correlation": corr_val,
                    "direction": direction,
                    "interpretation_pt": interpretation,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Dados filtrados
    # ------------------------------------------------------------------

    def get_filtered_data(self, filters: Dict) -> pd.DataFrame:
        """Aplica filtros ao dataset e retorna o subconjunto filtrado.

        Filtros suportados:

        - ``'gender'``: lista de gêneros, ex: ``['Male', 'Female']``
        - ``'age_min'``: idade mínima (inclusive)
        - ``'age_max'``: idade máxima (inclusive)
        - ``'obesity_classes'``: lista de classes de obesidade (valores em inglês)

        A coluna ``Obesity`` no DataFrame retornado é substituída pelo rótulo
        PT-BR correspondente de OBESITY_LABELS_PT.

        Args:
            filters: Dicionário com os filtros a aplicar.

        Returns:
            DataFrame filtrado com rótulos PT-BR na coluna ``Obesity``.
        """
        df = self.df.copy()

        gender_filter: Optional[List[str]] = filters.get("gender")
        if gender_filter:
            valid_genders = [
                g for g in gender_filter if g in CATEGORICAL_VALUES["Gender"]
            ]
            if valid_genders:
                df = df[df["Gender"].isin(valid_genders)]

        age_min: Optional[float] = filters.get("age_min")
        if age_min is not None:
            df = df[df["Age"] >= float(age_min)]

        age_max: Optional[float] = filters.get("age_max")
        if age_max is not None:
            df = df[df["Age"] <= float(age_max)]

        obesity_classes: Optional[List[str]] = filters.get("obesity_classes")
        if obesity_classes:
            valid_classes = [
                c for c in obesity_classes if c in CATEGORICAL_VALUES["Obesity"]
            ]
            if valid_classes:
                df = df[df["Obesity"].isin(valid_classes)]

        # Substituir rótulos EN por PT-BR na coluna Obesity
        df = df.copy()
        df["Obesity"] = df["Obesity"].map(OBESITY_LABELS_PT)

        logger.info(
            "get_filtered_data: %d registros após aplicar filtros %s",
            len(df),
            filters,
        )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _encode_for_correlation(self) -> pd.DataFrame:
        """Codifica o dataset para cálculo de correlação de Spearman.

        Returns:
            DataFrame com todas as features numéricas/codificadas e
            a coluna ``Obesity_encoded`` (ordinal 0–6).
        """
        df = self.df.copy()

        # Target ordinal
        df["Obesity_encoded"] = df["Obesity"].map(_OBESITY_ORDINAL)

        # Colunas numéricas nativas
        numeric_cols = list(NUMERIC_RANGES.keys())

        # Binárias → 0/1
        for col, mapping in _BINARY_COLS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Ordinais CAEC e CALC
        for col in ("CAEC", "CALC"):
            if col in df.columns:
                df[col] = df[col].map(_FREQ_ORDINAL)

        # Selecionar apenas as colunas de interesse (excluir MTRANS e Obesity original)
        feature_cols = (
            numeric_cols
            + list(_BINARY_COLS.keys())
            + ["CAEC", "CALC"]
        )
        # Filtrar apenas colunas que existem no DataFrame
        feature_cols = [c for c in feature_cols if c in df.columns]

        result = df[feature_cols + ["Obesity_encoded"]].dropna()
        return result
