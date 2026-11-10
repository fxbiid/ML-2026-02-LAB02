"""Experimentos de Olivetti y LFW con PCA, LDA y Fisherfaces."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.clasificadores.fabrica import FabricaClasificadores
from src.config import (
    DIMENSION_MAPAS,
    DIMENSIONES,
    DIR_RESULTADOS,
    RANDOM_STATE,
    TEST_SIZE,
    asegurar_directorios,
    ruta_modelo,
)
from src.datos.division import DivisorEstratificado
from src.datos.lfw import CargadorLFW
from src.datos.olivetti import CargadorOlivetti
from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fabrica import FabricaExtractores
from src.embeddings.pca import ExtractorPCA
from src.evaluacion.metricas import Evaluador
from src.evaluacion.visualizacion import Visualizador
from src.excepciones import EtapaPendienteAlumno
from src.modelos import ConjuntoRostros, Particion, ResultadoExperimento


class PipelineExperimentos:
    """Coordina la carga, el split, los embeddings y la evidencia en TEST."""

    def __init__(
        self,
        divisor: DivisorEstratificado | None = None,
        fabrica_extractores: FabricaExtractores | None = None,
        fabrica_clasificadores: FabricaClasificadores | None = None,
        evaluador: Evaluador | None = None,
        visualizador: Visualizador | None = None,
    ) -> None:
        self.divisor = divisor or DivisorEstratificado()
        self.fabrica_extractores = fabrica_extractores or FabricaExtractores()
        self.fabrica_clasificadores = fabrica_clasificadores or FabricaClasificadores()
        self.evaluador = evaluador or Evaluador()
        self.visualizador = visualizador or Visualizador()

    def ejecutar_olivetti(self) -> list[ResultadoExperimento]:
        print("== Experimento: Olivetti ==")
        return self._ejecutar(CargadorOlivetti().cargar())

    def ejecutar_lfw(self) -> list[ResultadoExperimento]:
        print("== Experimento: LFW ==")
        try:
            conjunto = CargadorLFW().cargar()
        except EtapaPendienteAlumno as exc:
            print(exc)
            return []
        return self._ejecutar(conjunto)

    def ejecutar_experimentos(self) -> None:
        self.ejecutar_olivetti()
        self.ejecutar_lfw()

    def _ejecutar(self, conjunto: ConjuntoRostros) -> list[ResultadoExperimento]:
        asegurar_directorios()
        particion = self.divisor.dividir(conjunto)
        print(
            f"X {tuple(conjunto.X.shape)}  "
            f"train {tuple(particion.X_train.shape)}  "
            f"test {tuple(particion.X_test.shape)}  "
            f"identidades {conjunto.n_clases}"
        )
        resultados: list[ResultadoExperimento] = []
        proyecciones_mapa: dict[tuple[str, int], tuple] = {}
        fisher_omitido = False

        for extractor in self.fabrica_extractores.para_experimento(conjunto.n_clases):
            if fisher_omitido and extractor.nombre == "fisherfaces":
                continue
            try:
                Z_train, Z_test = self._proyectar(extractor, particion)
            except EtapaPendienteAlumno as exc:
                print(exc)
                if extractor.nombre == "fisherfaces":
                    fisher_omitido = True
                continue

            if extractor.dimension == DIMENSION_MAPAS:
                proyecciones_mapa[(extractor.nombre, extractor.dimension)] = (
                    Z_train,
                    particion.y_train,
                )
            resultados.extend(
                self._evaluar_clasificadores(conjunto, extractor, particion, Z_train, Z_test)
            )

        self._figuras_pca(conjunto, particion)
        self._mapas(conjunto.nombre, proyecciones_mapa)
        self._guardar_csv(resultados, conjunto.nombre)
        self._imprimir(resultados)
        return resultados

    def _proyectar(
        self,
        extractor: ExtractorEmbedding,
        particion: Particion,
    ) -> tuple:
        extractor.fit(particion.X_train, particion.y_train)
        Z_train = extractor.transform(particion.X_train)
        Z_test = extractor.transform(particion.X_test)
        return Z_train, Z_test

    def _evaluar_clasificadores(
        self,
        conjunto: ConjuntoRostros,
        extractor: ExtractorEmbedding,
        particion: Particion,
        Z_train,
        Z_test,
    ) -> list[ResultadoExperimento]:
        filas: list[ResultadoExperimento] = []
        for nombre in self.fabrica_clasificadores.disponibles():
            modelo = self.fabrica_clasificadores.crear(nombre)
            medicion = self.evaluador.evaluar(
                modelo,
                Z_train,
                particion.y_train,
                Z_test,
                particion.y_test,
            )
            fila = ResultadoExperimento(
                dataset=conjunto.nombre,
                embedding=extractor.nombre,
                d=extractor.dimension,
                modelo=nombre,
                parametros=self.fabrica_clasificadores.parametros(nombre),
                accuracy=medicion.accuracy,
                macro_f1=medicion.macro_f1,
                tiempo_s=medicion.tiempo_s,
            )
            filas.append(fila)
            self._guardar_modelo(conjunto, extractor, nombre, modelo)
            ruta = self.visualizador.confusion(
                medicion.matriz,
                conjunto.nombre,
                extractor.nombre,
                nombre,
                extractor.dimension,
            )
            print(
                f"  {extractor.nombre:12} d={extractor.dimension:<3} {nombre:6} "
                f"acc={fila.accuracy:.3f}  macro-F1={fila.macro_f1:.3f}  "
                f"{fila.tiempo_s:.2f}s  -> {ruta.name}"
            )
        return filas

    def _guardar_modelo(self, conjunto, extractor, nombre: str, modelo) -> Path:
        ruta = ruta_modelo(conjunto.nombre, extractor.nombre, nombre, extractor.dimension)
        joblib.dump(
            {
                "extractor": extractor,
                "clasificador": modelo,
                "dataset": conjunto.nombre,
                "embedding": extractor.nombre,
                "modelo": nombre,
                "d": extractor.dimension,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "nombres": conjunto.nombres_identidades,
                "alto": conjunto.alto,
                "ancho": conjunto.ancho,
            },
            ruta,
        )
        return ruta

    def _figuras_pca(self, conjunto: ConjuntoRostros, particion: Particion) -> None:
        if conjunto.n_clases < 2:
            return
        d_ref = min(max(DIMENSIONES), particion.X_train.shape[0] - 1, particion.X_train.shape[1])
        referencia = ExtractorPCA(n_components=d_ref)
        referencia.fit(particion.X_train)
        self.visualizador.curva_pca(particion.X_train, conjunto.nombre)
        self.visualizador.eigenfaces(
            referencia.rostro_medio(conjunto.alto, conjunto.ancho),
            referencia.componentes_imagen(conjunto.alto, conjunto.ancho),
            conjunto.nombre,
        )

    def _mapas(self, dataset: str, proyecciones: dict) -> None:
        for (embedding, d), (Z_train, y_train) in proyecciones.items():
            for metodo in ("tsne", "umap"):
                ruta = self.visualizador.mapa(Z_train, y_train, metodo, dataset, embedding, d)
                print(f"  mapa {ruta.name}")

    def _guardar_csv(self, filas: list[ResultadoExperimento], dataset: str) -> Path:
        columnas = [
            "dataset",
            "embedding",
            "d",
            "modelo",
            "parametros",
            "accuracy",
            "macro_f1",
            "tiempo_s",
        ]
        nuevo = pd.DataFrame([fila.__dict__ for fila in filas], columns=columnas)
        if not nuevo.empty:
            nuevo["accuracy"] = nuevo["accuracy"].round(6)
            nuevo["macro_f1"] = nuevo["macro_f1"].round(6)
            nuevo["tiempo_s"] = nuevo["tiempo_s"].round(4)
        ruta = DIR_RESULTADOS / "resultados_modelos.csv"
        if ruta.exists():
            previo = pd.read_csv(ruta)
            previo = previo[previo["dataset"] != dataset]
            nuevo = pd.concat([previo, nuevo], ignore_index=True)
        nuevo.to_csv(ruta, index=False)
        print(f"Tabla guardada en {ruta}")
        return ruta

    @staticmethod
    def _imprimir(filas: list[ResultadoExperimento]) -> None:
        if not filas:
            print("Sin filas nuevas en la tabla.")
            return
        tabla = pd.DataFrame([fila.__dict__ for fila in filas])
        with pd.option_context("display.max_colwidth", 48, "display.width", 120):
            print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
