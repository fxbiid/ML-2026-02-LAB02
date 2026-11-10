"""Construcción de PCA, LDA y Fisherfaces para el mismo protocolo."""

from __future__ import annotations

from src.config import DIMENSIONES
from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fisherfaces import ExtractorFisherfaces
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class FabricaExtractores:
    """Crea un extractor por nombre o la lista completa del experimento."""

    def crear(self, nombre: str, d: int, n_clases: int) -> ExtractorEmbedding:
        if nombre == "pca":
            return ExtractorPCA(n_components=d)
        if nombre == "lda":
            return ExtractorLDA(n_components=self.dimension_lda(d, n_clases))
        if nombre == "fisherfaces":
            return ExtractorFisherfaces(
                d_pca=d,
                d_lda=self.dimension_lda(d, n_clases),
            )
        raise ValueError(f"Embedding desconocido: {nombre}")

    def para_experimento(self, n_clases: int) -> list[ExtractorEmbedding]:
        """PCA, LDA y Fisherfaces en las dimensiones del protocolo."""
        extractores: list[ExtractorEmbedding] = []
        lda_vistos: set[int] = set()
        for d in DIMENSIONES:
            extractores.append(ExtractorPCA(n_components=d))
            d_lda = self.dimension_lda(d, n_clases)
            if d_lda not in lda_vistos:
                lda_vistos.add(d_lda)
                extractores.append(ExtractorLDA(n_components=d_lda))
            extractores.append(ExtractorFisherfaces(d_pca=d, d_lda=d_lda))
        return extractores

    @staticmethod
    def dimension_lda(d: int, n_clases: int) -> int:
        """LDA no puede superar C − 1 ni pedir más coordenadas que las de entrada."""
        return max(1, min(d, n_clases - 1))
