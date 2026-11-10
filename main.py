#!/usr/bin/env python3
"""Orquestador de experimentos (punto de entrada).

Uso (con el entorno conda activado):

    python main.py olivetti       # Experimento con Olivetti
    python main.py lfw            # Experimento con LFW
    python main.py experimentos   # Ambos conjuntos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite `python main.py` sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import PipelineExperimentos  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimentos de embeddings faciales lineales (LAB02).",
    )
    parser.add_argument(
        "etapa",
        choices=["olivetti", "lfw", "experimentos"],
        help="Conjunto a evaluar. 'experimentos' evalúa Olivetti y LFW.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    pipeline = PipelineExperimentos()
    acciones = {
        "olivetti": pipeline.ejecutar_olivetti,
        "lfw": pipeline.ejecutar_lfw,
        "experimentos": pipeline.ejecutar_experimentos,
    }
    acciones[args.etapa]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
