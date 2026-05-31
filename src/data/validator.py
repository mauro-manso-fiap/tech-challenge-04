"""
Validação de schema e ranges do dataset de obesidade.

Verifica colunas esperadas, ranges numéricos e valores categóricos válidos
antes de qualquer processamento downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.config import CATEGORICAL_VALUES, EXPECTED_COLUMNS, NUMERIC_RANGES
from src.logging_config import setup_logging

logger: logging.Logger = setup_logging(__name__)


@dataclass
class ValidationResult:
    """Resultado da validação de um DataFrame ou registro individual.

    Attributes:
        is_valid: True se nenhum erro foi encontrado.
        errors: Lista de mensagens de erro descritivas com row numbers quando aplicável.
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)


class DataValidator:
    """Valida schema, ranges numéricos e valores categóricos do dataset de obesidade.

    Usa as constantes definidas em ``src/config.py`` (EXPECTED_COLUMNS,
    NUMERIC_RANGES, CATEGORICAL_VALUES) para que nenhum valor fique hard-coded
    neste módulo.

    Example::

        validator = DataValidator()
        result = validator.validate(df)
        if not result.is_valid:
            for err in result.errors:
                print(err)
    """

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Valida um DataFrame completo contra o schema esperado.

        Verifica em ordem:
        1. Presença das 17 colunas esperadas.
        2. Ranges numéricos para cada coluna em NUMERIC_RANGES.
        3. Valores categóricos para cada coluna em CATEGORICAL_VALUES.

        Cada erro é logado com ``logging.WARNING`` e inclui o número da linha
        (índice do DataFrame) onde o problema foi encontrado.

        Args:
            df: DataFrame pandas a ser validado.

        Returns:
            ValidationResult com ``is_valid=True`` se não houver erros,
            ou ``is_valid=False`` com a lista de erros encontrados.
        """
        errors: List[str] = []

        # --- 1. Verificar colunas esperadas ---
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        for col in missing_cols:
            msg = f"Missing column: '{col}'"
            logger.warning(msg)
            errors.append(msg)

        # Se colunas estão faltando, não faz sentido continuar validando valores
        if missing_cols:
            return ValidationResult(is_valid=False, errors=errors)

        # --- 2. Verificar ranges numéricos ---
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if col not in df.columns:
                continue  # já capturado acima
            out_of_range = df[~df[col].between(min_val, max_val, inclusive="both")]
            for row_idx, value in out_of_range[col].items():
                msg = (
                    f"Row {row_idx}: {col} value {value} is out of range "
                    f"[{min_val}, {max_val}]"
                )
                logger.warning(msg)
                errors.append(msg)

        # --- 3. Verificar valores categóricos ---
        for col, valid_values in CATEGORICAL_VALUES.items():
            if col not in df.columns:
                continue  # já capturado acima
            invalid_mask = ~df[col].isin(valid_values)
            invalid_rows = df[invalid_mask]
            for row_idx, value in invalid_rows[col].items():
                msg = (
                    f"Row {row_idx}: {col} value '{value}' is not in "
                    f"valid values {valid_values}"
                )
                logger.warning(msg)
                errors.append(msg)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_single_record(self, record: Dict) -> ValidationResult:
        """Valida um único registro (dicionário) para uso em predição.

        Verifica ranges numéricos e valores categóricos sem exigir a coluna
        alvo ``Obesity`` (que não está disponível em tempo de inferência).
        Não inclui row numbers nas mensagens de erro pois é um registro único.

        Args:
            record: Dicionário com os campos do paciente. Deve conter as 16
                features de entrada (sem a coluna ``Obesity``).

        Returns:
            ValidationResult com ``is_valid=True`` se não houver erros,
            ou ``is_valid=False`` com a lista de erros encontrados.
        """
        errors: List[str] = []

        # Colunas de entrada esperadas (sem a coluna alvo 'Obesity')
        input_columns = [col for col in EXPECTED_COLUMNS if col != "Obesity"]
        missing_cols = [col for col in input_columns if col not in record]
        for col in missing_cols:
            msg = f"Missing field: '{col}'"
            logger.warning(msg)
            errors.append(msg)

        # --- Verificar ranges numéricos ---
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if col not in record:
                continue  # já capturado acima
            value = record[col]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                msg = f"{col} value '{value}' is not numeric"
                logger.warning(msg)
                errors.append(msg)
                continue

            if not (min_val <= numeric_value <= max_val):
                msg = (
                    f"{col} value {numeric_value} is out of range "
                    f"[{min_val}, {max_val}]"
                )
                logger.warning(msg)
                errors.append(msg)

        # --- Verificar valores categóricos (exceto 'Obesity') ---
        categorical_input = {
            col: vals
            for col, vals in CATEGORICAL_VALUES.items()
            if col != "Obesity"
        }
        for col, valid_values in categorical_input.items():
            if col not in record:
                continue  # já capturado acima
            value = record[col]
            if value not in valid_values:
                msg = (
                    f"{col} value '{value}' is not in valid values {valid_values}"
                )
                logger.warning(msg)
                errors.append(msg)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
