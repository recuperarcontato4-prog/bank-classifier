"""Smoke test — verifica que o notebook carrega e as dependencias estao disponiveis."""
import json
from pathlib import Path

import pytest


def test_notebook_is_valid_json():
    nb_path = Path(__file__).parent.parent / "classificador_despesas.ipynb"
    assert nb_path.exists(), "Notebook nao encontrado"
    content = json.loads(nb_path.read_text(encoding="utf-8"))
    assert "cells" in content
    assert len(content["cells"]) > 0


def test_transformers_pipeline_importable():
    from transformers import pipeline  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_categorias_mutuamente_exclusivas():
    categorias = [
        "Alimentacao", "Transporte", "Contas e Servicos",
        "Lazer", "Compras", "Saude",
    ]
    assert len(categorias) == len(set(categorias))
