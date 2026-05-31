"""
Configuração centralizada de logging para o pipeline de analytics de obesidade.

Uso:
    from src.logging_config import setup_logging

    logger = setup_logging(__name__)
    logger.info("Mensagem informativa")
    logger.warning("Aviso importante")
    logger.error("Erro crítico")
"""

import logging
import sys


def setup_logging(name: str = __name__) -> logging.Logger:
    """Configura e retorna um logger com nível INFO e formato padrão.

    O handler é adicionado apenas uma vez para evitar duplicação de mensagens
    em chamadas repetidas com o mesmo nome de logger.

    Args:
        name: Nome do logger, tipicamente ``__name__`` do módulo chamador.

    Returns:
        Logger configurado com StreamHandler para stdout.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Evita propagação para o root logger (evita duplicatas)
        logger.propagate = False

    return logger
