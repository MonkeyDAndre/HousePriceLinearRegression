"""Estrae ed esegue le celle di codice taggate di un notebook Jupytext."""
from pathlib import Path

import jupytext


def carica_namespace(percorso: Path, tag: tuple[str, ...] = ("imports", "helpers")) -> dict:
    notebook = jupytext.read(percorso)
    # __name__ non importabile: cloudpickle (usato da n_jobs=-1) serializza le funzioni per valore
    namespace: dict = {"__name__": "notebook_sotto_test"}
    for cella in notebook.cells:
        tag_cella = set(cella.metadata.get("tags", []))
        if cella.cell_type == "code" and tag_cella & set(tag):
            exec(compile(cella.source, str(percorso), "exec"), namespace)
    return namespace
