"""
Serviço de predição de obesidade.

Carrega o pipeline treinado (pipeline.joblib) e os metadados do modelo
(model_metadata.json) na inicialização, expondo um método ``predict``
que valida o input, aplica o pipeline e retorna um ``PredictionResult``
com classe predita, probabilidades, IMC e nível de risco.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import pandas as pd

from src.config import OBESITY_LABELS_PT, RESULT_INFO
from src.data.validator import DataValidator
from src.logging_config import setup_logging

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Mapeamento de classe → nível de risco
# ---------------------------------------------------------------------------

_RISK_LEVEL_MAP: Dict[str, str] = {
    "Insufficient_Weight": "baixo",
    "Normal_Weight": "baixo",
    "Overweight_Level_I": "moderado",
    "Overweight_Level_II": "moderado",
    "Obesity_Type_I": "alto",
    "Obesity_Type_II": "muito_alto",
    "Obesity_Type_III": "muito_alto",
}

# Caminhos padrão relativos à raiz do projeto
_DEFAULT_PIPELINE_PATH = os.path.join("models", "pipeline.joblib")
_DEFAULT_METADATA_PATH = os.path.join("models", "model_metadata.json")


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Resultado completo de uma predição de obesidade.

    Attributes:
        predicted_class: Nome interno da classe predita (ex: "Obesity_Type_I").
        predicted_label_pt: Rótulo em PT-BR da classe predita (ex: "Obesidade Tipo I").
        confidence: Probabilidade da classe predita (0.0–1.0).
        all_probabilities: Dicionário {nome_classe: probabilidade} para as 7 classes.
        bmi: IMC calculado como peso / altura².
        risk_level: Nível de risco — "baixo", "moderado", "alto" ou "muito_alto".
        model_version: Versão do modelo lida do model_metadata.json.
        training_date: Data de treinamento lida do model_metadata.json.
        accuracy: Acurácia no test set lida do model_metadata.json.
    """

    predicted_class: str
    predicted_label_pt: str
    confidence: float
    all_probabilities: Dict[str, float]
    bmi: float
    risk_level: str
    model_version: str
    training_date: str
    accuracy: float


# ---------------------------------------------------------------------------
# Serviço de predição
# ---------------------------------------------------------------------------


class PredictionService:
    """Carrega o pipeline treinado e serve predições de obesidade.

    O pipeline (``pipeline.joblib``) inclui o ``FeatureEngineer`` e o
    estimador, garantindo paridade treino/inferência. Os metadados do
    modelo (versão, data, acurácia) são lidos de ``model_metadata.json``.

    Example::

        service = PredictionService()
        result = service.predict({
            "Gender": "Male", "Age": 25.0, "Height": 1.75, "Weight": 70.0,
            "family_history": "yes", "FAVC": "yes", "FCVC": 2.0, "NCP": 3.0,
            "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.0, "SCC": "no",
            "FAF": 1.0, "TUE": 1.0, "CALC": "no",
            "MTRANS": "Public_Transportation",
        })
        print(result.predicted_label_pt, result.confidence)
    """

    def __init__(
        self,
        pipeline_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> None:
        """Inicializa o serviço carregando o pipeline e os metadados.

        Args:
            pipeline_path: Caminho para ``pipeline.joblib``. Se ``None``,
                usa ``models/pipeline.joblib`` relativo ao diretório de
                trabalho atual.
            metadata_path: Caminho para ``model_metadata.json``. Se ``None``,
                usa ``models/model_metadata.json`` relativo ao diretório de
                trabalho atual.

        Raises:
            FileNotFoundError: Se ``pipeline.joblib`` não for encontrado no
                caminho especificado.
        """
        resolved_pipeline = pipeline_path or _DEFAULT_PIPELINE_PATH
        resolved_metadata = metadata_path or _DEFAULT_METADATA_PATH

        if not os.path.exists(resolved_pipeline):
            raise FileNotFoundError(
                f"Pipeline não encontrado em '{resolved_pipeline}'. "
                "Execute o notebook 03_model_training.ipynb para gerar o arquivo."
            )

        self._pipeline = joblib.load(resolved_pipeline)
        logger.info("Pipeline carregado de '%s'", resolved_pipeline)

        # Metadados do modelo
        self._model_version: str = "N/A"
        self._training_date: str = "N/A"
        self._accuracy: float = 0.0

        if os.path.exists(resolved_metadata):
            with open(resolved_metadata, "r", encoding="utf-8") as fh:
                metadata: Dict = json.load(fh)
            self._model_version = str(metadata.get("model_version", "N/A"))
            self._training_date = str(metadata.get("training_date", "N/A"))
            self._accuracy = float(metadata.get("accuracy", 0.0))
            logger.info(
                "Metadados carregados — versão: %s | data: %s | acurácia: %.4f",
                self._model_version,
                self._training_date,
                self._accuracy,
            )
        else:
            logger.warning(
                "Arquivo de metadados não encontrado em '%s'. "
                "Usando valores padrão (N/A).",
                resolved_metadata,
            )

        self._validator = DataValidator()

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, input_data: Dict) -> PredictionResult:
        """Valida o input, aplica o pipeline e retorna o resultado da predição.

        Args:
            input_data: Dicionário com os 16 campos de entrada do paciente
                (sem a coluna ``Obesity``). Exemplo::

                    {
                        "Gender": "Male", "Age": 25.0, "Height": 1.75,
                        "Weight": 70.0, "family_history": "yes", ...
                    }

        Returns:
            ``PredictionResult`` com classe predita, probabilidades, IMC,
            nível de risco e metadados do modelo.

        Raises:
            ValueError: Se o input falhar na validação de schema/ranges/categorias.
        """
        # --- 1. Validação ---
        validation = self._validator.validate_single_record(input_data)
        if not validation.is_valid:
            error_summary = "; ".join(validation.errors)
            raise ValueError(f"Input inválido: {error_summary}")

        # --- 2. Converter para DataFrame de uma linha ---
        df = pd.DataFrame([input_data])

        # --- 3. Predição ---
        probabilities_array = self._pipeline.predict_proba(df)
        predicted_class_array = self._pipeline.predict(df)

        predicted_class: str = str(predicted_class_array[0])
        class_names = list(self._pipeline.classes_)
        proba_row = probabilities_array[0]

        confidence: float = float(proba_row[class_names.index(predicted_class)])
        all_probabilities: Dict[str, float] = {
            cls: float(prob) for cls, prob in zip(class_names, proba_row)
        }

        # --- 4. IMC ---
        height = float(input_data["Height"])
        weight = float(input_data["Weight"])
        bmi: float = weight / (height ** 2)

        # --- 5. Nível de risco ---
        risk_level: str = _RISK_LEVEL_MAP.get(predicted_class, "moderado")

        # --- 6. Rótulo PT-BR ---
        predicted_label_pt: str = OBESITY_LABELS_PT.get(predicted_class, predicted_class)

        result = PredictionResult(
            predicted_class=predicted_class,
            predicted_label_pt=predicted_label_pt,
            confidence=confidence,
            all_probabilities=all_probabilities,
            bmi=round(bmi, 2),
            risk_level=risk_level,
            model_version=self._model_version,
            training_date=self._training_date,
            accuracy=self._accuracy,
        )

        logger.info(
            "Predição: %s (%.1f%%) | IMC: %.2f | Risco: %s",
            predicted_class,
            confidence * 100,
            bmi,
            risk_level,
        )

        return result
