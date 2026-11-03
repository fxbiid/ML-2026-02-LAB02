# Laboratorio: embeddings faciales lineales

Pipeline académico para comparar **PCA** y **LDA** en el reconocimiento de rostros, y dejar listo el lugar donde el alumno completa **Fisherfaces** y el experimento con **LFW**.

La pregunta de trabajo es si una fotografía nueva corresponde a una identidad ya conocida. El píxel no se compara directo: primero se obtiene un embedding lineal y después un clasificador elige la identidad.

## Requisitos

- [Conda](https://docs.conda.io/) (Miniconda o Anaconda)
- Uso académico de Olivetti y LFW, con atribución a scikit-learn y a las fuentes originales de las fotografías

## Entorno conda

No use `pip install` ni `requirements.txt`. El entorno oficial es `environment.yml`.

```bash
conda env create -f environment.yml
conda activate lab-embeddings-faciales
```

Para recrearlo:

```bash
conda env update -f environment.yml --prune
```

Los valores por defecto (semilla 42, 25 % de prueba, caché en `data/cache`) alcanzan para ejecutar el laboratorio. Si quiere cambiarlos:

```bash
cp .env.example .env
```

## Cómo ejecutar

Desde la raíz del repositorio, con el entorno activado:

```bash
python main.py olivetti       # PCA y LDA; guarda tabla, figuras y modelos
python main.py lfw            # TODO(alumno): mismo protocolo cuando complete el cargador
python main.py experimentos   # Olivetti e intento de LFW
streamlit run app.py          # rostros de prueba con PCA o LDA
```

La primera ejecución descarga Olivetti en `data/cache/`. TRAIN aprende la media, las direcciones y el clasificador. TEST solo pasa por `transform` y `predict`.

Salida esperada al cargar Olivetti: `X (400, 4096)`, train `(300, 4096)`, test `(100, 4096)`.

## Qué está implementado y qué debe completar

| Módulo | Estado | Qué hace |
| --- | --- | --- |
| `src/datos/olivetti.py` | Listo | 400 fotos, 40 identidades, 64×64 |
| `src/datos/division.py` | Listo | Split estratificado, índices disjuntos |
| `src/embeddings/pca.py` | Listo | PCA ajustado solo con TRAIN; eigenfaces y varianza |
| `src/embeddings/lda.py` | Listo | LDA con `solver="svd"`; `d ≤ C−1` |
| `src/clasificadores/` | Listo | SVM lineal, k-NN (k=3) y árbol (`max_depth=8`) |
| `src/evaluacion/` | Listo | Accuracy, macro-F1, confusión, curva PCA, t-SNE y UMAP |
| `src/interfaz/` y `app.py` | Listo | Foto de prueba, acierto y probabilidad |
| `src/embeddings/fisherfaces.py` | Listo | PCA y después LDA, ambos ajustados en TRAIN |
| `src/datos/lfw.py` | `TODO(alumno)` | Hasta 50 identidades con al menos 20 fotos |

Si ejecuta una etapa pendiente, el programa imprime una pista y **no falla en silencio**. Cuando Fisherfaces deje de lanzar `EtapaPendienteAlumno`, el mismo bucle lo evalúa con SVM, k-NN y árbol. Cuando `CargadorLFW.cargar` devuelva un `ConjuntoRostros`, `python main.py lfw` reutiliza ese protocolo.

## Protocolo

- Semilla `42` y `test_size=0.25`, con `stratify=y`.
- Dimensiones informadas: `d = 10, 20, 30`. LDA queda acotada a `C − 1`.
- Clasificadores fijos, todos con `predict_proba`:
  - SVM: `kernel=linear`, `C=1.0`, `probability=True`. La clase predicha sale del margen; la probabilidad de la interfaz es el escalado de Platt. scikit-learn 1.9 depreca ese argumento, y la fábrica silencia solo ese aviso: el reemplazo `CalibratedClassifierCV` cambia la identidad elegida.
  - k-NN: 3 vecinos; la probabilidad es la fracción de votos
  - Árbol: profundidad máxima 8; la probabilidad es la proporción de la hoja
- La tabla no elige un ganador con el conjunto de prueba. Muestra las combinaciones para compararlas.
- t-SNE (`perplexity=30`) y UMAP (`n_neighbors=15`, `min_dist=0.1`, `random_state=42`) se calculan sobre `Z_train` con `d=20`. Un mapa 2D no reemplaza la accuracy.

`data/resultados/resultados_modelos.csv` guarda dataset, embedding, `d`, modelo, parámetros, accuracy, macro-F1 y el tiempo de ajuste más predicción del clasificador. Las figuras quedan en `figuras/`. Los modelos, en `data/modelos/`. Esas salidas se regeneran y no se suben a Git.

## Interfaz

`streamlit run app.py` muestra una fotografía del conjunto de prueba de Olivetti. En la barra se elige PCA o LDA, el clasificador y `d`. La pantalla indica si la identidad predicha coincide con la real y con qué probabilidad.

No cargue fotografías personales. Los rostros son datos biométricos.

## Presentación

La presentación de la sesión está en `docs/lab-embeddings-faciales.tex`, con el tema Auriga. Esa carpeta es material docente local: está en `.gitignore` y no se publica en GitHub.

```bash
cd docs
pdflatex lab-embeddings-faciales.tex
pdflatex lab-embeddings-faciales.tex
```

## Diseño en breve

`PipelineExperimentos` coordina:

1. `CargadorOlivetti` (o `CargadorLFW`, cuando el alumno lo complete).
2. `DivisorEstratificado`, que deja cada identidad de test también en train.
3. `FabricaExtractores`: PCA, LDA y el gancho de Fisherfaces.
4. `FabricaClasificadores` y `Evaluador` sobre el embedding, nunca sobre los píxeles de test para ajustar.
5. `Visualizador` para la curva de varianza, las eigenfaces, las confusiones y los mapas.
6. `ServicioReconocimiento`, que lee el mismo artefacto que guardó el orquestador.

## Ética

- El análisis es académico y no justifica vigilancia ni un rendimiento en poblaciones nuevas.
- Use Olivetti y LFW respetando sus condiciones y cite las fuentes.
- No incorpore fotografías personales sin autorización ni infiera atributos sensibles.
- Un acierto en este conjunto de prueba no demuestra que el sistema reconozca a otras personas.

## Estructura del repositorio

```
main.py                 Experimentos, tabla y figuras
app.py                  Interfaz Streamlit
environment.yml         Entorno conda
src/pipeline.py         PipelineExperimentos
src/datos/              Olivetti listo; LFW pendiente
src/embeddings/         PCA y LDA listos; Fisherfaces pendiente
src/clasificadores/     SVM, k-NN y árbol
src/evaluacion/         Métricas y figuras
src/interfaz/           Predicción sobre el conjunto de prueba
data/                   Caché, modelos y CSV (generados)
figuras/                Curva, eigenfaces, mapas y confusiones
docs/                   Presentación Beamer (solo copia local)
```
