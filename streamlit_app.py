"""
streamlit_app.py — ponto de entrada alternativo para o Streamlit Cloud (raiz do projeto).

Este arquivo permite fazer deploy a partir da raiz do repositório, caso o campo
"Main file path" seja configurado como `streamlit_app.py` em vez de
`apps/prediction_app.py`.

O app real está em apps/prediction_app.py. Este arquivo apenas ajusta o sys.path
e executa o conteúdo do app de predição.

Uso recomendado:
    - Main file path: apps/prediction_app.py  (preferencial)
    - Main file path: streamlit_app.py         (alternativo, usa este arquivo)
"""

import os
import sys

# Garante que a raiz do projeto está no sys.path para importações de src/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Executa o app de predição como se fosse o arquivo principal
_app_path = os.path.join(PROJECT_ROOT, "apps", "prediction_app.py")
with open(_app_path, encoding="utf-8") as _f:
    exec(_f.read())  # noqa: S102
