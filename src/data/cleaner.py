"""
Módulo de limpeza de dados para o pipeline de analytics de obesidade.

Responsabilidades:
- Remoção de duplicatas
- Imputação de valores ausentes (mediana para numéricos, moda para categóricos)
- Padronização de case em colunas categóricas
"""

from typing import Dict, Any

import pandas as pd

from src.config import CATEGORICAL_VALUES, NUMERIC_RANGES
from src.logging_config import setup_logging

logger = setup_logging(__name__)


class DataCleaner:
    """Limpa e padroniza o DataFrame do dataset de obesidade.

    Operações realizadas em ordem:
    1. Remove linhas duplicadas
    2. Imputa valores ausentes em colunas numéricas com a mediana
    3. Imputa valores ausentes em colunas categóricas com a moda
    4. Padroniza o case das colunas categóricas conforme ``CATEGORICAL_VALUES``
    """

    # Colunas numéricas derivadas de NUMERIC_RANGES (sem hard-code)
    _NUMERIC_COLS: list[str] = list(NUMERIC_RANGES.keys())

    # Colunas categóricas derivadas de CATEGORICAL_VALUES (sem hard-code)
    _CATEGORICAL_COLS: list[str] = list(CATEGORICAL_VALUES.keys())

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas, imputa missing values e padroniza categóricos.

        Trabalha em uma cópia do DataFrame original — não modifica in-place.

        Args:
            df: DataFrame bruto carregado do CSV.

        Returns:
            DataFrame limpo e padronizado.
        """
        result = df.copy()

        logger.info("Iniciando limpeza. Shape inicial: %s", result.shape)

        result = self._remove_duplicates(result)
        result = self._impute_numeric(result)
        result = self._impute_categorical(result)
        result = self._standardize_categorical_case(result)

        logger.info("Limpeza concluída. Shape final: %s", result.shape)

        return result

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linhas duplicadas e loga a contagem removida."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        logger.info(
            "Duplicatas removidas: %d (de %d → %d linhas)", n_removed, n_before, len(df)
        )
        return df

    def _impute_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa valores ausentes em colunas numéricas com a mediana."""
        for col in self._NUMERIC_COLS:
            if col not in df.columns:
                continue
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    "Coluna '%s': %d valor(es) ausente(s) imputado(s) com mediana=%.4f",
                    col,
                    n_missing,
                    median_val,
                )
        return df

    def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputa valores ausentes em colunas categóricas com a moda."""
        for col in self._CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                mode_val = df[col].mode(dropna=True)
                if mode_val.empty:
                    logger.info(
                        "Coluna '%s': %d valor(es) ausente(s), mas moda não pôde ser calculada.",
                        col,
                        n_missing,
                    )
                    continue
                mode_val = mode_val.iloc[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(
                    "Coluna '%s': %d valor(es) ausente(s) imputado(s) com moda='%s'",
                    col,
                    n_missing,
                    mode_val,
                )
        return df

    def _standardize_categorical_case(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza o case das colunas categóricas conforme CATEGORICAL_VALUES.

        Para cada coluna categórica, constrói um mapeamento case-insensitive
        dos valores válidos definidos em ``CATEGORICAL_VALUES`` e aplica a
        substituição. Valores que não correspondem a nenhum valor válido são
        mantidos inalterados (serão detectados pelo DataValidator).
        """
        for col, valid_values in CATEGORICAL_VALUES.items():
            if col not in df.columns:
                continue

            # Mapeamento: versão lower-stripped → valor canônico
            canonical_map: Dict[str, str] = {v.strip().lower(): v for v in valid_values}

            def _normalize(val: Any) -> Any:
                if pd.isna(val):
                    return val
                normalized = str(val).strip().lower()
                return canonical_map.get(normalized, val)

            original_series = df[col].copy()
            df[col] = df[col].map(_normalize)

            changed = (df[col] != original_series).sum()
            if changed > 0:
                logger.info(
                    "Coluna '%s': %d valor(es) padronizado(s) para case canônico.",
                    col,
                    changed,
                )

        return df


def get_cleaning_report(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
) -> Dict[str, Any]:
    """Gera um relatório resumido comparando o DataFrame original e o limpo.

    Args:
        df_original: DataFrame antes da limpeza.
        df_cleaned: DataFrame após a limpeza.

    Returns:
        Dicionário com as seguintes chaves:
        - ``duplicates_removed``: número de linhas duplicadas removidas.
        - ``missing_imputed``: dict ``{coluna: n_valores_imputados}`` para
          colunas que tinham valores ausentes no original.
        - ``shape_before``: tupla ``(linhas, colunas)`` do original.
        - ``shape_after``: tupla ``(linhas, colunas)`` do limpo.
    """
    duplicates_removed: int = len(df_original) - df_original.drop_duplicates().shape[0]

    missing_imputed: Dict[str, int] = {}
    for col in df_original.columns:
        n_missing = int(df_original[col].isna().sum())
        if n_missing > 0:
            missing_imputed[col] = n_missing

    report: Dict[str, Any] = {
        "duplicates_removed": duplicates_removed,
        "missing_imputed": missing_imputed,
        "shape_before": df_original.shape,
        "shape_after": df_cleaned.shape,
    }

    logger.info(
        "Relatório de limpeza: %d duplicata(s) removida(s), %d coluna(s) com imputação, "
        "shape %s → %s",
        duplicates_removed,
        len(missing_imputed),
        df_original.shape,
        df_cleaned.shape,
    )

    return report
