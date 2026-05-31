"""
Feature engineering para o pipeline de analytics de obesidade.

Implementa ``FeatureEngineer`` como um transformador sklearn compatível com
``Pipeline``, ``GridSearchCV`` e serialização via ``joblib``.

O ``fit`` aprende os parâmetros de scaling **apenas** nos dados de treino.
O ``transform`` aplica todas as transformações usando os parâmetros aprendidos,
garantindo paridade treino/inferência e evitando data leakage.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import CATEGORICAL_VALUES
from src.logging_config import setup_logging

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Constantes de encoding (derivadas de CATEGORICAL_VALUES — sem hard-code)
# ---------------------------------------------------------------------------

# Mapeamento ordinal para CAEC e CALC
_ORDINAL_MAP: dict[str, int] = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3,
}

# Categorias válidas de MTRANS (ordem determinística para one-hot)
_MTRANS_CATEGORIES: list[str] = sorted(CATEGORICAL_VALUES["MTRANS"])

# Colunas one-hot geradas para MTRANS
_MTRANS_OHE_COLS: list[str] = [f"MTRANS_{cat}" for cat in _MTRANS_CATEGORIES]

# Colunas escaladas com StandardScaler
_STANDARD_COLS: list[str] = ["Age", "Height", "Weight", "BMI"]

# Colunas escaladas com MinMaxScaler
_MINMAX_COLS: list[str] = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Valor máximo possível de eating_score (para normalização [0, 1])
# eating_score = (FAVC_binary * 2) + FCVC + NCP + CAEC_ordinal
# max = (1 * 2) + 3 + 4 + 3 = 12
_EATING_SCORE_MAX: float = 12.0

# Limites para clipping do lifestyle_score antes da normalização
_LIFESTYLE_CLIP_MIN: float = -7.0   # FAF=0, TUE=2, CALC=3, SMOKE=1 → 0-2-(3*0.5)-(1*2) = -5.5
_LIFESTYLE_CLIP_MAX: float = 3.0    # FAF=3, TUE=0, CALC=0, SMOKE=0 → 3-0-0-0 = 3


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforma o dataset de obesidade em features prontas para modelagem.

    Implementa a interface sklearn ``BaseEstimator`` + ``TransformerMixin``
    para ser embutido em um ``sklearn.Pipeline``:

    .. code-block:: python

        pipeline = Pipeline([
            ("fe", FeatureEngineer()),
            ("clf", RandomForestClassifier()),
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, "models/pipeline.joblib")

    O ``fit`` aprende os parâmetros de ``StandardScaler`` e ``MinMaxScaler``
    **apenas** nos dados de treino. O ``transform`` aplica todas as
    transformações usando esses parâmetros, sem refitá-los.

    Transformações aplicadas em ``transform`` (nesta ordem):
    1. BMI = Weight / Height²
    2. eating_score (FAVC, FCVC, NCP, CAEC) normalizado para [0, 1]
    3. lifestyle_score (FAF, TUE, CALC, SMOKE) normalizado para [0, 1]
    4. high_calorie_sedentary = (FAVC == "yes") & (FAF < 1) → int
    5. Label encoding binário: Gender, FAVC, SMOKE, SCC, family_history → 0/1
    6. Ordinal encoding: CAEC, CALC → no=0, Sometimes=1, Frequently=2, Always=3
    7. One-hot encoding: MTRANS → 5 colunas binárias
    8. StandardScaler: Age, Height, Weight, BMI
    9. MinMaxScaler: FCVC, NCP, CH2O, FAF, TUE

    Attributes:
        standard_scaler_: StandardScaler ajustado em ``fit``.
        minmax_scaler_: MinMaxScaler ajustado em ``fit``.
        is_fitted_: bool indicando se ``fit`` foi chamado.
    """

    def __init__(self) -> None:
        """Inicializa o FeatureEngineer com scalers não ajustados."""
        self.standard_scaler_: StandardScaler = StandardScaler()
        self.minmax_scaler_: MinMaxScaler = MinMaxScaler()
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Interface sklearn
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Aprende parâmetros de scaling nos dados de treino.

        Calcula as features intermediárias (BMI, eating_score, lifestyle_score)
        e ajusta ``StandardScaler`` e ``MinMaxScaler`` nos dados de treino.
        Não modifica ``X``.

        Args:
            X: DataFrame com as features de entrada (sem a coluna alvo).
            y: Ignorado. Presente apenas para compatibilidade com a API sklearn.

        Returns:
            self — permite encadeamento ``fit(X).transform(X)``.

        Raises:
            ValueError: Se alguma coluna obrigatória estiver ausente em ``X``.
        """
        logger.info("FeatureEngineer.fit iniciado. Shape: %s", X.shape)

        self._validate_input_columns(X)

        # Trabalha em cópia para não modificar X
        X_fit = X.copy()

        # Calcula BMI para ajustar o StandardScaler com a coluna BMI
        X_fit = self._compute_bmi(X_fit)

        # Ajusta StandardScaler em Age, Height, Weight, BMI
        self.standard_scaler_.fit(X_fit[_STANDARD_COLS])
        logger.info(
            "StandardScaler ajustado em colunas: %s", _STANDARD_COLS
        )

        # Ajusta MinMaxScaler em FCVC, NCP, CH2O, FAF, TUE
        self.minmax_scaler_.fit(X_fit[_MINMAX_COLS])
        logger.info(
            "MinMaxScaler ajustado em colunas: %s", _MINMAX_COLS
        )

        self.is_fitted_ = True
        logger.info("FeatureEngineer.fit concluído.")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica todas as transformações de feature engineering.

        Usa os parâmetros aprendidos em ``fit`` — não reajusta os scalers.
        Trabalha em uma cópia de ``X`` para não modificar o DataFrame original.

        Args:
            X: DataFrame com as features de entrada (sem a coluna alvo).

        Returns:
            Array numpy 2D com todas as features transformadas, na ordem
            retornada por ``get_feature_names_out()``.

        Raises:
            ValueError: Se ``fit`` não foi chamado antes de ``transform``,
                ou se alguma coluna obrigatória estiver ausente.
        """
        if not self.is_fitted_:
            raise ValueError(
                "FeatureEngineer não foi ajustado. Chame fit() antes de transform()."
            )

        logger.info("FeatureEngineer.transform iniciado. Shape: %s", X.shape)

        self._validate_input_columns(X)

        # Trabalha em cópia para não modificar X
        df = X.copy()

        # --- 1. BMI = Weight / Height² ---
        df = self._compute_bmi(df)

        # --- 2. eating_score ---
        df = self._compute_eating_score(df)

        # --- 3. lifestyle_score ---
        df = self._compute_lifestyle_score(df)

        # --- 4. high_calorie_sedentary ---
        df = self._compute_high_calorie_sedentary(df)

        # --- 5. Label encoding binários ---
        df = self._apply_label_encoding(df)

        # --- 6. Ordinal encoding CAEC e CALC ---
        df = self._apply_ordinal_encoding(df)

        # --- 7. One-hot encoding MTRANS ---
        df = self._apply_ohe_mtrans(df)

        # --- 8. StandardScaler: Age, Height, Weight, BMI ---
        df[_STANDARD_COLS] = self.standard_scaler_.transform(df[_STANDARD_COLS])

        # --- 9. MinMaxScaler: FCVC, NCP, CH2O, FAF, TUE ---
        df[_MINMAX_COLS] = self.minmax_scaler_.transform(df[_MINMAX_COLS])

        # Seleciona e ordena colunas de saída
        output_cols = self.get_feature_names_out()
        result = df[output_cols].to_numpy(dtype=np.float64)

        logger.info(
            "FeatureEngineer.transform concluído. Shape de saída: %s", result.shape
        )

        return result

    def get_feature_names_out(self) -> List[str]:
        """Retorna a lista ordenada de nomes das features de saída.

        A ordem corresponde às colunas do array retornado por ``transform``.

        Returns:
            Lista de strings com os nomes das features na ordem de saída.
        """
        return [
            # Features numéricas escaladas com StandardScaler
            "Age",
            "Height",
            "Weight",
            "BMI",
            # Features numéricas escaladas com MinMaxScaler
            "FCVC",
            "NCP",
            "CH2O",
            "FAF",
            "TUE",
            # Features engineered
            "eating_score",
            "lifestyle_score",
            "high_calorie_sedentary",
            # Label encoding binário
            "Gender",
            "FAVC",
            "SMOKE",
            "SCC",
            "family_history",
            # Ordinal encoding
            "CAEC",
            "CALC",
            # One-hot encoding MTRANS
            *_MTRANS_OHE_COLS,
        ]

    # ------------------------------------------------------------------
    # Métodos internos de transformação
    # ------------------------------------------------------------------

    def _compute_bmi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula BMI = Weight / Height².

        Args:
            df: DataFrame com colunas ``Weight`` e ``Height``.

        Returns:
            DataFrame com coluna ``BMI`` adicionada.

        Raises:
            ValueError: Se algum valor de ``Height`` for ≤ 0.
        """
        if (df["Height"] <= 0).any():
            raise ValueError(
                "Coluna 'Height' contém valores ≤ 0. BMI não pode ser calculado."
            )
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        logger.info("BMI calculado. Média: %.4f", df["BMI"].mean())
        return df

    def _compute_eating_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula eating_score normalizado para [0, 1].

        Fórmula:
            FAVC_binary = 1 se FAVC == "yes", else 0
            CAEC_ordinal = no=0, Sometimes=1, Frequently=2, Always=3
            raw = (FAVC_binary * 2) + FCVC + NCP + CAEC_ordinal
            eating_score = raw / _EATING_SCORE_MAX

        Args:
            df: DataFrame com colunas ``FAVC``, ``FCVC``, ``NCP``, ``CAEC``.

        Returns:
            DataFrame com coluna ``eating_score`` adicionada.
        """
        favc_binary = (df["FAVC"].str.lower() == "yes").astype(float)
        caec_ordinal = df["CAEC"].map(_ORDINAL_MAP).astype(float)

        raw = (favc_binary * 2.0) + df["FCVC"].astype(float) + df["NCP"].astype(float) + caec_ordinal
        df["eating_score"] = raw / _EATING_SCORE_MAX

        logger.info(
            "eating_score calculado. Média: %.4f, Min: %.4f, Max: %.4f",
            df["eating_score"].mean(),
            df["eating_score"].min(),
            df["eating_score"].max(),
        )
        return df

    def _compute_lifestyle_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula lifestyle_score normalizado para [0, 1].

        Fórmula:
            CALC_ordinal = no=0, Sometimes=1, Frequently=2, Always=3
            SMOKE_binary = 1 se SMOKE == "yes", else 0
            raw = FAF - TUE - (CALC_ordinal * 0.5) - (SMOKE_binary * 2)
            clipped = clip(raw, _LIFESTYLE_CLIP_MIN, _LIFESTYLE_CLIP_MAX)
            lifestyle_score = (clipped - _LIFESTYLE_CLIP_MIN) /
                              (_LIFESTYLE_CLIP_MAX - _LIFESTYLE_CLIP_MIN)

        Args:
            df: DataFrame com colunas ``FAF``, ``TUE``, ``CALC``, ``SMOKE``.

        Returns:
            DataFrame com coluna ``lifestyle_score`` adicionada.
        """
        calc_ordinal = df["CALC"].map(_ORDINAL_MAP).astype(float)
        smoke_binary = (df["SMOKE"].str.lower() == "yes").astype(float)

        raw = (
            df["FAF"].astype(float)
            - df["TUE"].astype(float)
            - (calc_ordinal * 0.5)
            - (smoke_binary * 2.0)
        )

        clipped = raw.clip(lower=_LIFESTYLE_CLIP_MIN, upper=_LIFESTYLE_CLIP_MAX)
        score_range = _LIFESTYLE_CLIP_MAX - _LIFESTYLE_CLIP_MIN
        df["lifestyle_score"] = (clipped - _LIFESTYLE_CLIP_MIN) / score_range

        logger.info(
            "lifestyle_score calculado. Média: %.4f, Min: %.4f, Max: %.4f",
            df["lifestyle_score"].mean(),
            df["lifestyle_score"].min(),
            df["lifestyle_score"].max(),
        )
        return df

    def _compute_high_calorie_sedentary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria flag binária high_calorie_sedentary.

        Valor 1 se FAVC == "yes" E FAF < 1, caso contrário 0.

        Args:
            df: DataFrame com colunas ``FAVC`` e ``FAF``.

        Returns:
            DataFrame com coluna ``high_calorie_sedentary`` adicionada.
        """
        favc_yes = df["FAVC"].str.lower() == "yes"
        faf_low = df["FAF"].astype(float) < 1.0
        df["high_calorie_sedentary"] = (favc_yes & faf_low).astype(int)

        n_flagged = df["high_calorie_sedentary"].sum()
        logger.info(
            "high_calorie_sedentary: %d registro(s) flagado(s) (%.1f%%)",
            n_flagged,
            100.0 * n_flagged / len(df),
        )
        return df

    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica label encoding binário (0/1) nas colunas categóricas binárias.

        Mapeamentos:
        - Gender: Male=1, Female=0
        - FAVC, SMOKE, SCC, family_history: yes=1, no=0

        Args:
            df: DataFrame com as colunas binárias originais.

        Returns:
            DataFrame com as colunas substituídas por valores 0/1.
        """
        # Gender: Male=1, Female=0
        df["Gender"] = (df["Gender"].str.lower() == "male").astype(int)

        # yes/no → 1/0
        for col in ("FAVC", "SMOKE", "SCC", "family_history"):
            df[col] = (df[col].str.lower() == "yes").astype(int)

        logger.info("Label encoding binário aplicado em: Gender, FAVC, SMOKE, SCC, family_history")
        return df

    def _apply_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica ordinal encoding em CAEC e CALC.

        Mapeamento: no=0, Sometimes=1, Frequently=2, Always=3

        Args:
            df: DataFrame com colunas ``CAEC`` e ``CALC`` como strings.

        Returns:
            DataFrame com ``CAEC`` e ``CALC`` substituídas por inteiros.
        """
        df["CAEC"] = df["CAEC"].map(_ORDINAL_MAP).astype(int)
        df["CALC"] = df["CALC"].map(_ORDINAL_MAP).astype(int)

        logger.info("Ordinal encoding aplicado em: CAEC, CALC")
        return df

    def _apply_ohe_mtrans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica one-hot encoding em MTRANS, gerando 5 colunas binárias.

        Colunas geradas (ordem alfabética):
        MTRANS_Automobile, MTRANS_Bike, MTRANS_Motorbike,
        MTRANS_Public_Transportation, MTRANS_Walking

        A coluna original ``MTRANS`` é removida.

        Args:
            df: DataFrame com coluna ``MTRANS`` como string.

        Returns:
            DataFrame com ``MTRANS`` removida e 5 colunas OHE adicionadas.
        """
        for cat in _MTRANS_CATEGORIES:
            col_name = f"MTRANS_{cat}"
            df[col_name] = (df["MTRANS"] == cat).astype(int)

        df = df.drop(columns=["MTRANS"])

        logger.info(
            "One-hot encoding aplicado em MTRANS. Colunas geradas: %s",
            _MTRANS_OHE_COLS,
        )
        return df

    # ------------------------------------------------------------------
    # Validação interna
    # ------------------------------------------------------------------

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        """Verifica que todas as colunas de entrada obrigatórias estão presentes.

        Args:
            X: DataFrame a ser validado.

        Raises:
            ValueError: Se alguma coluna obrigatória estiver ausente.
        """
        required = {
            "Gender", "Age", "Height", "Weight", "family_history",
            "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O",
            "SCC", "FAF", "TUE", "CALC", "MTRANS",
        }
        missing = required - set(X.columns)
        if missing:
            raise ValueError(
                f"Colunas obrigatórias ausentes no DataFrame de entrada: {sorted(missing)}"
            )
