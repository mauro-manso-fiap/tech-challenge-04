"""
Módulo de avaliação de modelos de classificação de obesidade.

Fornece funções para calcular métricas de desempenho e gerar visualizações
diagnósticas: matriz de confusão, curvas ROC, importância de features e
correlação de Spearman.

Uso:
    from src.models.evaluator import evaluate, plot_confusion_matrix

    result = evaluate(pipeline, X_test, y_test)
    fig = plot_confusion_matrix(result.confusion_matrix, result.class_names)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

from src.config import CATEGORICAL_VALUES, OBESITY_LABELS_PT
from src.logging_config import setup_logging

logger = setup_logging(__name__)

# Ordem canônica das 7 classes (mesma do config.py)
_CLASS_ORDER: List[str] = CATEGORICAL_VALUES["Obesity"]

# Rótulos PT-BR na mesma ordem canônica
_CLASS_LABELS_PT: List[str] = [OBESITY_LABELS_PT[c] for c in _CLASS_ORDER]


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Resultado completo da avaliação de um pipeline de classificação.

    Attributes:
        accuracy: Acurácia global no conjunto de teste.
        macro_f1: F1-score macro-médio.
        per_class_metrics: Dicionário {nome_classe: {precision, recall, f1}}.
        confusion_matrix: Matriz de confusão 7×7 (numpy array).
        class_names: Lista ordenada dos nomes das classes (inglês, dataset).
    """

    accuracy: float
    macro_f1: float
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((7, 7), dtype=int)
    )
    class_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Avaliação principal
# ---------------------------------------------------------------------------


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Avalia um pipeline sklearn no conjunto de teste.

    Calcula acurácia, macro-F1, precision/recall/F1 por classe e a matriz de
    confusão 7×7. Todos os resultados são logados em nível INFO.

    Args:
        pipeline: Pipeline sklearn já treinado (com FeatureEngineer + estimador).
        X_test: Features do conjunto de teste.
        y_test: Rótulos verdadeiros do conjunto de teste.

    Returns:
        EvaluationResult com todas as métricas calculadas.
    """
    y_pred = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Determina as classes presentes (mantém ordem canônica)
    present_classes = [c for c in _CLASS_ORDER if c in y_test.unique()]

    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=present_classes,
        average=None,
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for cls, prec, rec, f1 in zip(present_classes, precision_arr, recall_arr, f1_arr):
        per_class[cls] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    cm = confusion_matrix(y_test, y_pred, labels=present_classes)

    # --- Logging ---
    logger.info("=== Avaliação do Modelo ===")
    logger.info("Acurácia: %.4f", accuracy)
    logger.info("F1-Score Macro: %.4f", macro_f1)
    logger.info("Métricas por classe:")
    for cls, metrics in per_class.items():
        label_pt = OBESITY_LABELS_PT.get(cls, cls)
        logger.info(
            "  %-30s | Precision: %.4f | Recall: %.4f | F1: %.4f",
            label_pt,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        class_names=present_classes,
    )


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusão",
) -> matplotlib.figure.Figure:
    """Plota a matriz de confusão como heatmap 7×7.

    Args:
        cm: Matriz de confusão (numpy array quadrado).
        class_names: Lista de nomes das classes na mesma ordem das linhas/colunas
            da matriz (nomes em inglês do dataset; serão traduzidos para PT-BR).
        title: Título do gráfico em PT-BR.

    Returns:
        Objeto ``matplotlib.figure.Figure`` com o heatmap. Não chama plt.show().
    """
    # Traduz para PT-BR
    labels_pt = [OBESITY_LABELS_PT.get(c, c) for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_pt,
        yticklabels=labels_pt,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Classe Predita", fontsize=11)
    ax.set_ylabel("Classe Real", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    logger.info("Matriz de confusão gerada (%dx%d).", cm.shape[0], cm.shape[1])
    return fig


# ---------------------------------------------------------------------------
# Curvas ROC
# ---------------------------------------------------------------------------


def plot_roc_curves(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
) -> matplotlib.figure.Figure:
    """Plota curvas ROC one-vs-rest para cada uma das 7 classes.

    Requer que o estimador do pipeline suporte ``predict_proba``.

    Args:
        pipeline: Pipeline sklearn treinado.
        X_test: Features do conjunto de teste.
        y_test: Rótulos verdadeiros.
        class_names: Lista de nomes das classes (inglês, dataset).

    Returns:
        Objeto ``matplotlib.figure.Figure`` com todas as curvas ROC.
    """
    y_prob = pipeline.predict_proba(X_test)

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_test)
    # LabelBinarizer ordena as classes alfabeticamente; alinhamos com class_names
    lb_classes: List[str] = list(lb.classes_)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))  # type: ignore[attr-defined]

    for i, cls in enumerate(class_names):
        label_pt = OBESITY_LABELS_PT.get(cls, cls)
        if cls not in lb_classes:
            logger.warning("Classe '%s' não encontrada nas classes do LabelBinarizer.", cls)
            continue

        cls_idx_lb = lb_classes.index(cls)
        cls_idx_prob = list(pipeline.classes_).index(cls) if hasattr(pipeline, "classes_") else i

        # Obtém o índice correto nas probabilidades (ordem do pipeline)
        try:
            pipeline_classes = list(pipeline.classes_)
            cls_idx_prob = pipeline_classes.index(cls)
        except (AttributeError, ValueError):
            # Fallback: tenta pelo estimador final
            try:
                clf = pipeline.named_steps.get("clf") or pipeline[-1]
                pipeline_classes = list(clf.classes_)
                cls_idx_prob = pipeline_classes.index(cls)
            except (AttributeError, ValueError, KeyError):
                cls_idx_prob = i

        y_true_bin = y_bin[:, cls_idx_lb]
        y_score = y_prob[:, cls_idx_prob]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        auc = roc_auc_score(y_true_bin, y_score)

        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{label_pt} (AUC = {auc:.3f})")
        logger.info("ROC AUC — %s: %.4f", label_pt, auc)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Classificador Aleatório")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Curvas ROC — One-vs-Rest (7 Classes)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=11)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Importância de features
# ---------------------------------------------------------------------------


def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
) -> matplotlib.figure.Figure:
    """Plota as top-N features mais importantes do modelo.

    Suporta:
    - RandomForest / XGBoost: usa ``feature_importances_``
    - MLP: usa a soma dos valores absolutos dos pesos da primeira camada

    Args:
        pipeline: Pipeline sklearn treinado.
        feature_names: Lista de nomes das features na ordem usada pelo estimador.
        top_n: Número de features a exibir.

    Returns:
        Objeto ``matplotlib.figure.Figure`` com o gráfico de barras horizontal.
    """
    # Extrai o estimador final do pipeline
    try:
        clf = pipeline.named_steps.get("clf") or pipeline[-1]
    except (AttributeError, KeyError):
        clf = pipeline

    importances: np.ndarray | None = None

    if hasattr(clf, "feature_importances_"):
        importances = np.array(clf.feature_importances_)
        logger.info("Importância de features extraída via feature_importances_.")
    elif hasattr(clf, "coefs_"):
        # MLP: soma dos valores absolutos dos pesos da primeira camada (input → hidden[0])
        importances = np.abs(clf.coefs_[0]).sum(axis=1)
        logger.info("Importância de features extraída via pesos da primeira camada (MLP).")
    else:
        logger.warning(
            "O estimador '%s' não possui feature_importances_ nem coefs_. "
            "Não é possível gerar o gráfico de importância.",
            type(clf).__name__,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Importância de features não disponível\npara este tipo de modelo.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_title("Importância de Features", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    # Alinha com feature_names (pode haver discrepância de tamanho)
    n_features = min(len(importances), len(feature_names))
    importances = importances[:n_features]
    names = feature_names[:n_features]

    # Seleciona top_n
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [names[i] for i in indices]

    # Inverte para exibir a mais importante no topo
    top_importances = top_importances[::-1]
    top_names = top_names[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    bars = ax.barh(range(len(top_names)), top_importances, color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_title(
        f"Importância das Top {len(top_names)} Features",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Importância Relativa", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    # Adiciona valores nas barras
    for bar, val in zip(bars, top_importances):
        ax.text(
            bar.get_width() + max(top_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    logger.info("Gráfico de importância de features gerado (top %d).", len(top_names))
    return fig


# ---------------------------------------------------------------------------
# Correlação de Spearman
# ---------------------------------------------------------------------------


def plot_spearman_correlation(
    df: pd.DataFrame,
    target: str,
    title: str = "Correlação de Spearman",
) -> matplotlib.figure.Figure:
    """Plota a correlação de Spearman entre features numéricas e o target.

    Se o target for categórico (string), é codificado ordinalmente usando a
    ordem canônica de ``OBESITY_LABELS_PT`` (Insufficient_Weight → Obesity_Type_III).

    Args:
        df: DataFrame com features e coluna target.
        target: Nome da coluna alvo.
        title: Título do gráfico em PT-BR.

    Returns:
        Objeto ``matplotlib.figure.Figure`` com o gráfico de barras horizontal.
    """
    df_work = df.copy()

    # Codifica o target se for categórico
    if df_work[target].dtype == object or str(df_work[target].dtype) == "category":
        ordinal_map = {cls: i for i, cls in enumerate(_CLASS_ORDER)}
        df_work[target] = df_work[target].map(ordinal_map)
        logger.info(
            "Target '%s' codificado ordinalmente para cálculo de Spearman.", target
        )

    # Seleciona apenas colunas numéricas (exceto o target)
    numeric_cols = [
        c
        for c in df_work.select_dtypes(include=[np.number]).columns
        if c != target
    ]

    if not numeric_cols:
        logger.warning("Nenhuma coluna numérica encontrada para correlação de Spearman.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Nenhuma feature numérica disponível.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    target_values = df_work[target].values

    correlations: Dict[str, float] = {}
    for col in numeric_cols:
        col_values = df_work[col].values
        # Remove NaN
        mask = ~(np.isnan(col_values) | np.isnan(target_values.astype(float)))
        if mask.sum() < 3:
            correlations[col] = 0.0
            continue
        rho, _ = stats.spearmanr(col_values[mask], target_values[mask])
        correlations[col] = float(rho) if not np.isnan(rho) else 0.0

    # Ordena por valor absoluto decrescente
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    feature_labels = [item[0] for item in sorted_items]
    rho_values = [item[1] for item in sorted_items]

    # Inverte para exibir maior correlação no topo
    feature_labels = feature_labels[::-1]
    rho_values = rho_values[::-1]

    colors = ["#2196F3" if v >= 0 else "#F44336" for v in rho_values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_labels) // 2)))
    ax.barh(range(len(feature_labels)), rho_values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Correlação de Spearman (ρ)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    # Legenda de cores
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", label="Correlação positiva"),
        Patch(facecolor="#F44336", label="Correlação negativa"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    logger.info("Gráfico de correlação de Spearman gerado (%d features).", len(feature_labels))
    return fig
