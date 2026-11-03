"""Fisherfaces: PCA comprime y LDA discrimina, aprendiendo solo de TRAIN."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    Orden obligatorio:

    1. Ajustar PCA con ``d_pca`` solo sobre ``X_train`` (ahí se estima μ).
    2. Ajustar LDA con ``d_lda`` sobre ese PCA de TRAIN, usando ``y_train``.
    3. Transformar TEST con las dos proyecciones, sin volver a ajustar.

    Contrato: ``d_lda <= min(C - 1, d_pca)``. Revise también el rango y
    la cantidad de muestras. Puede reutilizar ``ExtractorPCA`` y ``ExtractorLDA``.
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)
        self._pca = ExtractorPCA(n_components=self.d_pca)
        self._lda = ExtractorLDA(n_components=self.d_lda)

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita y_train: LDA usa las identidades.")
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Se esperan X de forma (N, p) e y de forma (N,).")
        if not np.isfinite(X).all():
            raise ValueError("Fisherfaces: X contiene valores no finitos.")

        n_muestras, n_pixeles = X.shape
        n_clases = np.unique(y).size
        if n_clases < 2 or n_muestras <= n_clases:
            raise ValueError("Fisherfaces requiere al menos dos clases y N > C.")
        maximo_pca = min(n_muestras - 1, n_pixeles)
        if self.d_pca > maximo_pca:
            raise ValueError(
                f"d_pca={self.d_pca} supera min(N-1, p)={maximo_pca}."
            )
        if self.d_lda > min(n_clases - 1, self.d_pca):
            raise ValueError(
                f"d_lda={self.d_lda} supera min(C-1, d_pca)="
                f"{min(n_clases - 1, self.d_pca)}."
            )

        # Ambos ajustes reciben exclusivamente TRAIN. PCA estima su media.
        pca = ExtractorPCA(n_components=self.d_pca)
        Z_pca = pca.fit_transform(X)
        singulares = pca._modelo.singular_values_
        tolerancia = (
            max(X.shape) * np.finfo(singulares.dtype).eps * singulares[0]
        )
        if np.count_nonzero(singulares > tolerancia) < self.d_pca:
            raise ValueError("d_pca supera el rango numérico de X centrada.")

        lda = ExtractorLDA(n_components=self.d_lda)
        # comprobar también detecta si el rango de LDA no alcanza d_lda.
        self.comprobar(lda.fit_transform(Z_pca, y), n_muestras)
        self._pca, self._lda = pca, lda
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TRAIN y TEST usan las mismas proyecciones; aquí nunca se llama a fit.
        Z_pca = self._pca.transform(X)
        return self.comprobar(self._lda.transform(Z_pca), X.shape[0])
