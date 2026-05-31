"""
Treinamento de modelos com otimização de hiperparâmetros via Optuna.

Implementa ``train_with_optuna`` para Random Forest, XGBoost e MLP,
usando ``TPESampler`` e ``StratifiedKFold`` dentro de cada trial.
O ``build_pipeline`` monta o ``sklearn.Pipeline`` final que é serializado
como ``pipeline.joblib``.

Fluxo:
    1. ``train_with_optuna`` otimiza hiperparâmetros via Optuna (f1_macro, CV-5)
    2. Melhor estimador é retreinado em X_train+y_train completo
    3. ``build_pipeline`` combina FeatureEngineer + estimador em um Pipeline
    4. ``compare_models`` seleciona o melhor entre os candidatos treinados
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.features.engineer import FeatureEngineer
from src.logging_config import setup_logging

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Suprimir logs verbosos do Optuna (manter apenas WARNING e acima)
# ---------------------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Enum de tipos de modelo
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """Tipos de modelo suportados pelo trainer."""

    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    MLP = "mlp"


# ---------------------------------------------------------------------------
# Dataclass de resultado de treinamento
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Resultado do treinamento de um modelo com Optuna.

    Attributes:
        model_type: Nome do tipo de modelo (valor do enum ModelType).
        best_params: Dicionário com os melhores hiperparâmetros encontrados.
        best_val_f1: F1-macro no validation set após retreino com melhores params.
        estimator: Estimador sklearn ajustado com os melhores hiperparâmetros.
        n_trials: Número de trials executados pelo Optuna.
    """

    model_type: str
    best_params: Dict[str, Any]
    best_val_f1: float
    estimator: Any
    n_trials: int = field(default=50)


# ---------------------------------------------------------------------------
# Funções internas de criação de estimadores
# ---------------------------------------------------------------------------


def _create_estimator(model_type: ModelType, params: Dict[str, Any]) -> Any:
    """Cria um estimador sklearn com os parâmetros fornecidos.

    Args:
        model_type: Tipo de modelo a ser criado.
        params: Dicionário de hiperparâmetros para o estimador.

    Returns:
        Estimador sklearn instanciado com os parâmetros fornecidos.

    Raises:
        ValueError: Se ``model_type`` não for um valor válido de ``ModelType``.
    """
    if model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == ModelType.XGBOOST:
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0,
            use_label_encoder=False,
        )
    elif model_type == ModelType.MLP:
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=500,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"ModelType desconhecido: {model_type}")


def _suggest_params(trial: optuna.Trial, model_type: ModelType) -> Dict[str, Any]:
    """Sugere hiperparâmetros para um trial Optuna.

    Args:
        trial: Trial Optuna atual.
        model_type: Tipo de modelo para o qual sugerir parâmetros.

    Returns:
        Dicionário com hiperparâmetros sugeridos pelo trial.

    Raises:
        ValueError: Se ``model_type`` não for um valor válido de ``ModelType``.
    """
    if model_type == ModelType.RANDOM_FOREST:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    elif model_type == ModelType.XGBOOST:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
    elif model_type == ModelType.MLP:
        hidden_layer_choices = [(64,), (128,), (64, 64), (128, 64), (256, 128)]
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [str(h) for h in hidden_layer_choices],
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
        }
    else:
        raise ValueError(f"ModelType desconhecido: {model_type}")


def _parse_mlp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Converte ``hidden_layer_sizes`` de string para tupla no MLP.

    O Optuna armazena ``hidden_layer_sizes`` como string (ex: ``"(64, 64)"``).
    Esta função converte de volta para tupla antes de instanciar o estimador.

    Args:
        params: Dicionário de parâmetros do MLP com ``hidden_layer_sizes`` como string.

    Returns:
        Dicionário com ``hidden_layer_sizes`` convertido para tupla de inteiros.
    """
    parsed = dict(params)
    if isinstance(parsed.get("hidden_layer_sizes"), str):
        # Converte "(64, 64)" → (64, 64)
        raw = parsed["hidden_layer_sizes"].strip("()")
        parsed["hidden_layer_sizes"] = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return parsed


# ---------------------------------------------------------------------------
# Função principal de treinamento
# ---------------------------------------------------------------------------


def train_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: ModelType,
    n_trials: int = 50,
) -> TrainingResult:
    """Treina um modelo com otimização de hiperparâmetros via Optuna.

    Cada trial cria um pipeline completo (FeatureEngineer + estimador) e
    avalia via ``StratifiedKFold(n_splits=5)`` em X_train/y_train, calculando
    a média do f1_macro entre os folds. Isso garante que o FeatureEngineer
    seja ajustado apenas nos dados de treino de cada fold (sem data leakage).

    Após a otimização, o melhor estimador é retreinado em X_train+y_train
    completo e avaliado em X_val/y_val para obter o ``best_val_f1`` final.

    Args:
        X_train: DataFrame com features de treino (raw, antes do FeatureEngineer).
        y_train: Series com rótulos de treino.
        X_val: DataFrame com features de validação (raw).
        y_val: Series com rótulos de validação.
        model_type: Tipo de modelo a ser treinado (``ModelType`` enum).
        n_trials: Número de trials Optuna. Padrão: 50.

    Returns:
        ``TrainingResult`` com os melhores parâmetros, f1_macro no validation
        set, estimador ajustado e número de trials executados.
    """
    logger.info(
        "Iniciando train_with_optuna | model_type=%s | n_trials=%d | "
        "X_train.shape=%s | X_val.shape=%s",
        model_type.value,
        n_trials,
        X_train.shape,
        X_val.shape,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        """Função objetivo do Optuna: f1_macro médio em CV-5."""
        params = _suggest_params(trial, model_type)

        # Para MLP, converter hidden_layer_sizes de string para tupla
        if model_type == ModelType.MLP:
            params = _parse_mlp_params(params)

        fold_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Pipeline fresco por fold para evitar data leakage
            estimator = _create_estimator(model_type, params)
            fold_pipeline = Pipeline([
                ("fe", FeatureEngineer()),
                ("clf", estimator),
            ])

            fold_pipeline.fit(X_fold_train, y_fold_train)
            y_pred = fold_pipeline.predict(X_fold_val)
            score = f1_score(y_fold_val, y_pred, average="macro", zero_division=0)
            fold_scores.append(score)

        mean_f1 = float(np.mean(fold_scores))
        return mean_f1

    # Criar e executar o estudo Optuna
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    logger.info(
        "Optuna concluído | model_type=%s | best_trial=%d | best_cv_f1=%.4f | params=%s",
        model_type.value,
        study.best_trial.number,
        study.best_value,
        best_params,
    )

    # Para MLP, converter hidden_layer_sizes de string para tupla nos best_params
    if model_type == ModelType.MLP:
        best_params = _parse_mlp_params(best_params)

    # Retreinar o melhor estimador em X_train+y_train completo
    best_estimator = _create_estimator(model_type, best_params)
    best_estimator.fit(
        FeatureEngineer().fit(X_train).transform(X_train),
        y_train,
    )

    # Avaliar no validation set
    fe_val = FeatureEngineer().fit(X_train).transform(X_val)
    y_val_pred = best_estimator.predict(fe_val)
    val_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))

    logger.info(
        "Retreino concluído | model_type=%s | val_f1_macro=%.4f",
        model_type.value,
        val_f1,
    )

    return TrainingResult(
        model_type=model_type.value,
        best_params=best_params,
        best_val_f1=val_f1,
        estimator=best_estimator,
        n_trials=n_trials,
    )


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_pipeline(feature_engineer: FeatureEngineer, estimator: Any) -> Pipeline:
    """Monta o sklearn Pipeline final para serialização.

    O pipeline combina o ``FeatureEngineer`` (já ajustado no treino) com o
    estimador (já ajustado no treino), formando o artefato ``pipeline.joblib``
    que é carregado pelo ``PredictionService`` em produção.

    Args:
        feature_engineer: Instância de ``FeatureEngineer`` já ajustada (fitted)
            nos dados de treino.
        estimator: Estimador sklearn já ajustado nos dados de treino transformados.

    Returns:
        ``sklearn.Pipeline`` com steps ``[("fe", feature_engineer), ("clf", estimator)]``.
    """
    pipeline = Pipeline([
        ("fe", feature_engineer),
        ("clf", estimator),
    ])
    logger.info(
        "Pipeline construído | steps=%s",
        [name for name, _ in pipeline.steps],
    )
    return pipeline


# ---------------------------------------------------------------------------
# Comparação de modelos
# ---------------------------------------------------------------------------


def compare_models(results: List[TrainingResult]) -> TrainingResult:
    """Compara múltiplos resultados de treinamento e retorna o melhor.

    Loga uma tabela de comparação com model_type, best_val_f1 e n_trials
    para todos os modelos, e retorna o ``TrainingResult`` com o maior
    ``best_val_f1``.

    Args:
        results: Lista de ``TrainingResult`` a serem comparados.

    Returns:
        O ``TrainingResult`` com o maior ``best_val_f1``.

    Raises:
        ValueError: Se ``results`` for uma lista vazia.
    """
    if not results:
        raise ValueError("A lista de resultados está vazia. Nenhum modelo para comparar.")

    logger.info("=" * 60)
    logger.info("Comparação de modelos:")
    logger.info("%-20s | %-12s | %-8s", "model_type", "val_f1_macro", "n_trials")
    logger.info("-" * 60)

    for r in sorted(results, key=lambda x: x.best_val_f1, reverse=True):
        logger.info(
            "%-20s | %-12.4f | %-8d",
            r.model_type,
            r.best_val_f1,
            r.n_trials,
        )

    logger.info("=" * 60)

    best = max(results, key=lambda r: r.best_val_f1)
    logger.info(
        "Melhor modelo: %s (val_f1_macro=%.4f)",
        best.model_type,
        best.best_val_f1,
    )
    return best
