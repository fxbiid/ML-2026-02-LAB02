"""Carga de LFW y selección de las identidades con más fotografías."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Hasta 50 identidades de LFW con al menos 20 fotos cada una."""

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        min_fotos = 20
        rostros = fetch_lfw_people(
            min_faces_per_person=min_fotos,
            resize=0.5,
            data_home=str(DIR_CACHE),
        )
        etiquetas = np.asarray(rostros.target, dtype=int)
        ids, frecuencias = np.unique(etiquetas, return_counts=True)
        elegibles = frecuencias >= min_fotos
        ids, frecuencias = ids[elegibles], frecuencias[elegibles]
        if ids.size < 2:
            raise ValueError("LFW requiere al menos dos identidades con 20 fotos cada una.")

        # Orden descendente por frecuencia y, en caso de empate, por ID original.
        orden = np.argsort(-frecuencias, kind="stable")
        conservar = np.isin(etiquetas, ids[orden[:50]])

        # La misma máscara conserva la correspondencia entre fotos y etiquetas.
        images = np.asarray(rostros.images)[conservar]
        X = np.asarray(rostros.data[conservar], dtype=float)
        ids_originales, y = np.unique(etiquetas[conservar], return_inverse=True)
        nombres = np.asarray(rostros.target_names)[ids_originales].tolist()
        _, alto, ancho = images.shape

        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        validar_conjunto(conjunto)
        return conjunto
