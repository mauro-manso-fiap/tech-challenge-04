"""
Testes pytest para src/data/validator.py.

Cobre: validação de schema, rejeição por range numérico e rejeição por
valor categórico inválido — conforme Requirement 12.1.

Validates: Requirements 12.1
"""

import copy

import pandas as pd
import pytest

from src.data.validator import DataValidator


# ---------------------------------------------------------------------------
# Fixture — DataFrame base válido (3 linhas com valores realistas)
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """DataFrame com 17 colunas e valores dentro dos ranges esperados."""
    data = {
        "Gender": ["Male", "Female", "Male"],
        "Age": [25.0, 32.0, 45.0],
        "Height": [1.75, 1.62, 1.80],
        "Weight": [70.0, 58.0, 90.0],
        "family_history": ["yes", "no", "yes"],
        "FAVC": ["yes", "no", "yes"],
        "FCVC": [2.0, 1.0, 3.0],
        "NCP": [3.0, 2.0, 4.0],
        "CAEC": ["Sometimes", "no", "Frequently"],
        "SMOKE": ["no", "no", "no"],
        "CH2O": [2.0, 1.0, 3.0],
        "SCC": ["no", "yes", "no"],
        "FAF": [1.0, 0.0, 2.0],
        "TUE": [1.0, 0.5, 2.0],
        "CALC": ["no", "Sometimes", "no"],
        "MTRANS": [
            "Public_Transportation",
            "Walking",
            "Automobile",
        ],
        "Obesity": ["Normal_Weight", "Insufficient_Weight", "Obesity_Type_I"],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestDataValidator:
    """Testes unitários para DataValidator."""

    def test_valid_dataframe_is_accepted(self, valid_df: pd.DataFrame) -> None:
        """DataFrame com 17 colunas corretas e valores válidos deve ser aceito."""
        validator = DataValidator()
        result = validator.validate(valid_df)

        assert result.is_valid is True
        assert result.errors == []

    def test_missing_column_is_rejected(self, valid_df: pd.DataFrame) -> None:
        """DataFrame sem a coluna 'Age' deve ser rejeitado com mensagem de erro."""
        df_missing = valid_df.drop(columns=["Age"])
        validator = DataValidator()
        result = validator.validate(df_missing)

        assert result.is_valid is False
        assert any("Age" in err for err in result.errors), (
            f"Esperava mensagem de erro mencionando 'Age', mas erros foram: {result.errors}"
        )

    def test_out_of_range_numeric_is_rejected(self, valid_df: pd.DataFrame) -> None:
        """DataFrame com Age=-1 (abaixo do mínimo 10) deve ser rejeitado."""
        df_bad = valid_df.copy()
        df_bad.loc[0, "Age"] = -1.0
        validator = DataValidator()
        result = validator.validate(df_bad)

        assert result.is_valid is False
        assert any("Age" in err for err in result.errors), (
            f"Esperava mensagem de erro mencionando 'Age', mas erros foram: {result.errors}"
        )

    def test_out_of_range_height_is_rejected(self, valid_df: pd.DataFrame) -> None:
        """DataFrame com Height=3.0 (acima do máximo 2.20) deve ser rejeitado."""
        df_bad = valid_df.copy()
        df_bad.loc[0, "Height"] = 3.0
        validator = DataValidator()
        result = validator.validate(df_bad)

        assert result.is_valid is False
        assert any("Height" in err for err in result.errors), (
            f"Esperava mensagem de erro mencionando 'Height', mas erros foram: {result.errors}"
        )

    def test_invalid_categorical_is_rejected(self, valid_df: pd.DataFrame) -> None:
        """DataFrame com Gender='Other' (não em ['Male','Female']) deve ser rejeitado."""
        df_bad = valid_df.copy()
        df_bad.loc[0, "Gender"] = "Other"
        validator = DataValidator()
        result = validator.validate(df_bad)

        assert result.is_valid is False
        assert any("Gender" in err for err in result.errors), (
            f"Esperava mensagem de erro mencionando 'Gender', mas erros foram: {result.errors}"
        )

    def test_validate_single_record_valid(self, valid_df: pd.DataFrame) -> None:
        """validate_single_record com registro válido deve retornar is_valid=True."""
        # Monta dicionário a partir da primeira linha do DataFrame base (sem 'Obesity')
        record = valid_df.iloc[0].drop("Obesity").to_dict()
        validator = DataValidator()
        result = validator.validate_single_record(record)

        assert result.is_valid is True
        assert result.errors == []

    def test_validate_single_record_invalid(self, valid_df: pd.DataFrame) -> None:
        """validate_single_record com Age=-1 deve retornar is_valid=False."""
        record = valid_df.iloc[0].drop("Obesity").to_dict()
        record["Age"] = -1.0
        validator = DataValidator()
        result = validator.validate_single_record(record)

        assert result.is_valid is False
        assert any("Age" in err for err in result.errors), (
            f"Esperava mensagem de erro mencionando 'Age', mas erros foram: {result.errors}"
        )
