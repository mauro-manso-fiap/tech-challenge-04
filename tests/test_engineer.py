"""
Testes pytest para FeatureEngineer (src/features/engineer.py).

Cobre:
- Invariante BMI: BMI = Weight / Height²
- Soma das colunas one-hot de MTRANS = 1 por registro
- Encoding ordinal de CAEC preserva ordem: no < Sometimes < Frequently < Always
- Sem data leakage: parâmetros do scaler aprendidos apenas no treino
- ValueError ao chamar transform() antes de fit()
- Shape de saída: 24 features
- get_feature_names_out() retorna lista correta de 24 strings

Validates: Requirements 12.2
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# Fixture — DataFrame base válido
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df() -> pd.DataFrame:
    """DataFrame com 5 linhas válidas e valores variados para testes."""
    return pd.DataFrame(
        {
            "Gender":          ["Male",   "Female", "Male",   "Female", "Male"],
            "Age":             [25.0,     30.0,     22.0,     45.0,     35.0],
            "Height":          [1.75,     1.60,     1.80,     1.65,     1.70],
            "Weight":          [70.0,     55.0,     90.0,     68.0,     80.0],
            "family_history":  ["yes",    "no",     "yes",    "no",     "yes"],
            "FAVC":            ["yes",    "no",     "yes",    "no",     "yes"],
            "FCVC":            [2.0,      1.0,      3.0,      2.0,      1.5],
            "NCP":             [3.0,      2.0,      4.0,      1.0,      2.0],
            "CAEC":            ["Sometimes", "no", "Frequently", "Always", "Sometimes"],
            "SMOKE":           ["no",     "no",     "yes",    "no",     "no"],
            "CH2O":            [2.0,      1.5,      3.0,      2.5,      1.0],
            "SCC":             ["no",     "yes",    "no",     "no",     "yes"],
            "FAF":             [1.0,      0.5,      2.0,      0.0,      1.5],
            "TUE":             [0.5,      1.0,      0.0,      2.0,      1.0],
            "CALC":            ["Sometimes", "no", "Frequently", "no", "Always"],
            "MTRANS":          [
                "Automobile",
                "Public_Transportation",
                "Walking",
                "Bike",
                "Motorbike",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mtrans_cols(feature_names: list[str]) -> list[str]:
    """Retorna apenas as colunas MTRANS_* da lista de features."""
    return [f for f in feature_names if f.startswith("MTRANS_")]


def _col_index(feature_names: list[str], col: str) -> int:
    return feature_names.index(col)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

class TestBMIInvariant:
    """Property 1 — BMI = Weight / Height² (Validates: Requirements 5.1, 12.2)."""

    def test_bmi_invariant(self, base_df: pd.DataFrame) -> None:
        """Para cada linha, o BMI antes do scaling deve ser Weight / Height²."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        result = fe.transform(base_df)

        feature_names = fe.get_feature_names_out()
        bmi_idx = _col_index(feature_names, "BMI")

        # O BMI no output está escalado pelo StandardScaler.
        # Recuperamos o valor original via inverse_transform.
        scaled_bmi = result[:, bmi_idx]
        # inverse_transform espera array 2D com todas as colunas standard
        std_cols = ["Age", "Height", "Weight", "BMI"]
        std_indices = [_col_index(feature_names, c) for c in std_cols]
        scaled_std = result[:, std_indices]
        original_std = fe.standard_scaler_.inverse_transform(scaled_std)
        bmi_col_in_std = std_cols.index("BMI")
        recovered_bmi = original_std[:, bmi_col_in_std]

        expected_bmi = base_df["Weight"].values / (base_df["Height"].values ** 2)

        for i, (recovered, expected) in enumerate(zip(recovered_bmi, expected_bmi)):
            assert abs(recovered - expected) < 0.001, (
                f"Linha {i}: BMI recuperado={recovered:.6f}, "
                f"esperado={expected:.6f} (diff={abs(recovered - expected):.6f})"
            )

    def test_bmi_raw_computation(self, base_df: pd.DataFrame) -> None:
        """Verifica diretamente que BMI = Weight / Height² para valores conhecidos."""
        # Linha com valores exatos para verificação determinística
        df = pd.DataFrame(
            {
                "Gender":         ["Male"],
                "Age":            [30.0],
                "Height":         [1.80],
                "Weight":         [81.0],   # BMI esperado = 81 / 1.80² = 25.0
                "family_history": ["no"],
                "FAVC":           ["no"],
                "FCVC":           [2.0],
                "NCP":            [3.0],
                "CAEC":           ["Sometimes"],
                "SMOKE":          ["no"],
                "CH2O":           [2.0],
                "SCC":            ["no"],
                "FAF":            [1.0],
                "TUE":            [0.5],
                "CALC":           ["no"],
                "MTRANS":         ["Walking"],
            }
        )
        fe = FeatureEngineer()
        fe.fit(df)
        result = fe.transform(df)

        feature_names = fe.get_feature_names_out()
        std_cols = ["Age", "Height", "Weight", "BMI"]
        std_indices = [_col_index(feature_names, c) for c in std_cols]
        scaled_std = result[:, std_indices]
        original_std = fe.standard_scaler_.inverse_transform(scaled_std)
        bmi_col_in_std = std_cols.index("BMI")
        recovered_bmi = original_std[0, bmi_col_in_std]

        expected_bmi = 81.0 / (1.80 ** 2)  # = 25.0
        assert abs(recovered_bmi - expected_bmi) < 0.001, (
            f"BMI esperado={expected_bmi:.4f}, recuperado={recovered_bmi:.4f}"
        )


class TestMtransOHESumsToOne:
    """Property 2 — Soma das colunas one-hot de MTRANS = 1 (Validates: Requirements 5.7, 12.2)."""

    def test_mtrans_ohe_sums_to_one(self, base_df: pd.DataFrame) -> None:
        """Para cada registro, a soma das colunas MTRANS_* deve ser exatamente 1."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        result = fe.transform(base_df)

        feature_names = fe.get_feature_names_out()
        mtrans_cols = _get_mtrans_cols(feature_names)
        assert len(mtrans_cols) == 5, f"Esperado 5 colunas MTRANS_*, encontrado {len(mtrans_cols)}"

        mtrans_indices = [_col_index(feature_names, c) for c in mtrans_cols]
        mtrans_values = result[:, mtrans_indices]
        row_sums = mtrans_values.sum(axis=1)

        for i, s in enumerate(row_sums):
            assert abs(s - 1.0) < 1e-9, (
                f"Linha {i}: soma das colunas MTRANS_* = {s:.6f}, esperado 1.0"
            )

    def test_mtrans_ohe_all_categories_covered(self) -> None:
        """Cada categoria de MTRANS gera exatamente uma coluna com valor 1."""
        from src.config import CATEGORICAL_VALUES

        all_mtrans = CATEGORICAL_VALUES["MTRANS"]
        rows = []
        for cat in all_mtrans:
            rows.append(
                {
                    "Gender": "Male", "Age": 25.0, "Height": 1.75, "Weight": 70.0,
                    "family_history": "no", "FAVC": "no", "FCVC": 2.0, "NCP": 3.0,
                    "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.0, "SCC": "no",
                    "FAF": 1.0, "TUE": 0.5, "CALC": "no", "MTRANS": cat,
                }
            )
        df = pd.DataFrame(rows)

        fe = FeatureEngineer()
        fe.fit(df)
        result = fe.transform(df)

        feature_names = fe.get_feature_names_out()
        mtrans_cols = _get_mtrans_cols(feature_names)
        mtrans_indices = [_col_index(feature_names, c) for c in mtrans_cols]
        mtrans_values = result[:, mtrans_indices]

        for i, cat in enumerate(all_mtrans):
            row = mtrans_values[i]
            assert row.sum() == 1.0, f"Categoria {cat}: soma = {row.sum()}"
            assert row.max() == 1.0, f"Categoria {cat}: nenhuma coluna com valor 1"


class TestCAECOrdinalOrder:
    """Property 3 — Ordinal encoding de CAEC preserva ordem (Validates: Requirements 5.6, 12.2)."""

    def test_caec_ordinal_order(self) -> None:
        """no < Sometimes < Frequently < Always após ordinal encoding."""
        caec_values = ["no", "Sometimes", "Frequently", "Always"]
        rows = []
        for caec in caec_values:
            rows.append(
                {
                    "Gender": "Male", "Age": 25.0, "Height": 1.75, "Weight": 70.0,
                    "family_history": "no", "FAVC": "no", "FCVC": 2.0, "NCP": 3.0,
                    "CAEC": caec, "SMOKE": "no", "CH2O": 2.0, "SCC": "no",
                    "FAF": 1.0, "TUE": 0.5, "CALC": "no", "MTRANS": "Walking",
                }
            )
        df = pd.DataFrame(rows)

        fe = FeatureEngineer()
        fe.fit(df)
        result = fe.transform(df)

        feature_names = fe.get_feature_names_out()
        caec_idx = _col_index(feature_names, "CAEC")
        caec_encoded = result[:, caec_idx]

        # Verifica ordem estrita: no < Sometimes < Frequently < Always
        assert caec_encoded[0] < caec_encoded[1], (
            f"'no' ({caec_encoded[0]}) deve ser < 'Sometimes' ({caec_encoded[1]})"
        )
        assert caec_encoded[1] < caec_encoded[2], (
            f"'Sometimes' ({caec_encoded[1]}) deve ser < 'Frequently' ({caec_encoded[2]})"
        )
        assert caec_encoded[2] < caec_encoded[3], (
            f"'Frequently' ({caec_encoded[2]}) deve ser < 'Always' ({caec_encoded[3]})"
        )

    def test_caec_ordinal_values_are_0_to_3(self) -> None:
        """Os valores ordinais de CAEC devem ser 0, 1, 2, 3 respectivamente."""
        caec_values = ["no", "Sometimes", "Frequently", "Always"]
        rows = []
        for caec in caec_values:
            rows.append(
                {
                    "Gender": "Male", "Age": 25.0, "Height": 1.75, "Weight": 70.0,
                    "family_history": "no", "FAVC": "no", "FCVC": 2.0, "NCP": 3.0,
                    "CAEC": caec, "SMOKE": "no", "CH2O": 2.0, "SCC": "no",
                    "FAF": 1.0, "TUE": 0.5, "CALC": "no", "MTRANS": "Walking",
                }
            )
        df = pd.DataFrame(rows)

        fe = FeatureEngineer()
        fe.fit(df)
        result = fe.transform(df)

        feature_names = fe.get_feature_names_out()
        caec_idx = _col_index(feature_names, "CAEC")
        caec_encoded = result[:, caec_idx]

        expected = [0, 1, 2, 3]
        for i, (enc, exp) in enumerate(zip(caec_encoded, expected)):
            assert enc == exp, (
                f"CAEC='{caec_values[i]}': esperado {exp}, obtido {enc}"
            )


class TestNoDataLeakage:
    """Property — fit no treino + transform no teste não vaza dados do teste
    (Validates: Requirements 5.8, 12.2)."""

    def test_no_data_leakage(self) -> None:
        """Parâmetros do StandardScaler devem refletir apenas os dados de treino."""
        # Treino: Age em torno de 25
        train_ages = [23.0, 24.0, 25.0, 26.0, 27.0]
        # Teste: Age em torno de 60 (bem diferente do treino)
        test_ages = [58.0, 59.0, 60.0, 61.0, 62.0]

        def _make_df(ages: list[float]) -> pd.DataFrame:
            n = len(ages)
            return pd.DataFrame(
                {
                    "Gender":         ["Male"] * n,
                    "Age":            ages,
                    "Height":         [1.75] * n,
                    "Weight":         [70.0] * n,
                    "family_history": ["no"] * n,
                    "FAVC":           ["no"] * n,
                    "FCVC":           [2.0] * n,
                    "NCP":            [3.0] * n,
                    "CAEC":           ["Sometimes"] * n,
                    "SMOKE":          ["no"] * n,
                    "CH2O":           [2.0] * n,
                    "SCC":            ["no"] * n,
                    "FAF":            [1.0] * n,
                    "TUE":            [0.5] * n,
                    "CALC":           ["no"] * n,
                    "MTRANS":         ["Walking"] * n,
                }
            )

        train_df = _make_df(train_ages)
        test_df = _make_df(test_ages)

        fe = FeatureEngineer()
        fe.fit(train_df)

        # Transforma ambos (não deve refitar o scaler)
        fe.transform(train_df)
        fe.transform(test_df)

        # O índice de Age no StandardScaler
        std_cols = ["Age", "Height", "Weight", "BMI"]
        age_idx_in_std = std_cols.index("Age")

        # mean_ do StandardScaler deve ser próximo de 25 (média do treino)
        scaler_age_mean = fe.standard_scaler_.mean_[age_idx_in_std]
        assert abs(scaler_age_mean - 25.0) < 1.0, (
            f"Média do StandardScaler para Age = {scaler_age_mean:.2f}; "
            f"esperado próximo de 25.0 (dados de treino). "
            f"Se estiver próximo de 60, há data leakage."
        )

        # Garante que a média NÃO é próxima de 60 (dados de teste)
        assert abs(scaler_age_mean - 60.0) > 10.0, (
            f"Média do StandardScaler para Age = {scaler_age_mean:.2f} está "
            f"próxima de 60.0 — indica data leakage dos dados de teste."
        )


class TestTransformWithoutFitRaises:
    """Verifica que transform() antes de fit() levanta ValueError."""

    def test_transform_without_fit_raises(self, base_df: pd.DataFrame) -> None:
        """Chamar transform() sem fit() deve levantar ValueError."""
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="fit"):
            fe.transform(base_df)


class TestOutputShape:
    """Verifica que o output tem exatamente 24 features."""

    def test_output_shape(self, base_df: pd.DataFrame) -> None:
        """O array retornado por transform() deve ter shape (n_rows, 24)."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        result = fe.transform(base_df)

        assert result.shape == (len(base_df), 24), (
            f"Shape esperado ({len(base_df)}, 24), obtido {result.shape}"
        )

    def test_output_shape_single_row(self) -> None:
        """Shape correto para um único registro."""
        df = pd.DataFrame(
            {
                "Gender": ["Female"], "Age": [28.0], "Height": [1.62],
                "Weight": [58.0], "family_history": ["yes"], "FAVC": ["yes"],
                "FCVC": [2.5], "NCP": [3.0], "CAEC": ["Frequently"],
                "SMOKE": ["no"], "CH2O": [2.0], "SCC": ["no"],
                "FAF": [0.5], "TUE": [1.0], "CALC": ["Sometimes"],
                "MTRANS": ["Public_Transportation"],
            }
        )
        fe = FeatureEngineer()
        fe.fit(df)
        result = fe.transform(df)

        assert result.shape == (1, 24), (
            f"Shape esperado (1, 24), obtido {result.shape}"
        )


class TestGetFeatureNamesOut:
    """Verifica que get_feature_names_out() retorna a lista correta de 24 strings."""

    def test_returns_24_features(self, base_df: pd.DataFrame) -> None:
        """get_feature_names_out() deve retornar exatamente 24 nomes."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        names = fe.get_feature_names_out()

        assert len(names) == 24, (
            f"Esperado 24 features, obtido {len(names)}: {names}"
        )

    def test_contains_required_features(self, base_df: pd.DataFrame) -> None:
        """A lista deve conter BMI, eating_score, lifestyle_score, high_calorie_sedentary."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        names = fe.get_feature_names_out()

        required = ["BMI", "eating_score", "lifestyle_score", "high_calorie_sedentary"]
        for feat in required:
            assert feat in names, f"Feature obrigatória '{feat}' ausente em get_feature_names_out()"

    def test_contains_all_mtrans_columns(self, base_df: pd.DataFrame) -> None:
        """A lista deve conter todas as 5 colunas MTRANS_*."""
        from src.config import CATEGORICAL_VALUES

        fe = FeatureEngineer()
        fe.fit(base_df)
        names = fe.get_feature_names_out()

        expected_mtrans = [f"MTRANS_{cat}" for cat in sorted(CATEGORICAL_VALUES["MTRANS"])]
        for col in expected_mtrans:
            assert col in names, f"Coluna one-hot '{col}' ausente em get_feature_names_out()"

    def test_all_names_are_strings(self, base_df: pd.DataFrame) -> None:
        """Todos os nomes retornados devem ser strings."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        names = fe.get_feature_names_out()

        for name in names:
            assert isinstance(name, str), f"Nome de feature não é string: {name!r}"

    def test_no_duplicate_names(self, base_df: pd.DataFrame) -> None:
        """Não deve haver nomes duplicados na lista de features."""
        fe = FeatureEngineer()
        fe.fit(base_df)
        names = fe.get_feature_names_out()

        assert len(names) == len(set(names)), (
            f"Nomes duplicados encontrados: {[n for n in names if names.count(n) > 1]}"
        )
