# Proyecto ENDIREH 2021: Análisis Exploratorio sobre Violencia contra las Mujeres

**Universidad Nacional Autónoma de México, Facultad de Ciencias**
Almacenes y Minería de Datos · Prácticas 3 y 4

---

## 1. Objetivo

Estructurar un proyecto de minería de datos con una arquitectura de carpetas
reproducible, seleccionar un framework de análisis, aplicar un preprocesamiento
riguroso y describir los datos mediante medidas de localización y variabilidad,
como etapas previas a cualquier modelado.

Sobre ese conjunto ya preprocesado, calcular e interpretar medidas de
heterogeneidad y de concentración, incluidos el coeficiente de Gini y la
entropía de Shannon, y comunicarlas con visualizaciones.

El caso de estudio es la Encuesta Nacional sobre la Dinámica de las Relaciones
en los Hogares (ENDIREH) 2021. Por tratarse de un tema social sensible, el
análisis distingue en todo momento entre la violencia **reportada** en la
encuesta y la ocurrencia del fenómeno, y evita atribuir causalidad a
asociaciones observadas en un diseño transversal.

## 2. Fuente de los datos

Los microdatos provienen de la **ENDIREH 2021**, levantada por el Instituto
Nacional de Estadística y Geografía (INEGI).

- Programa: <https://www.inegi.org.mx/programas/endireh/2021/>
- Descriptor de archivos (diccionario de variables):
  <https://www.inegi.org.mx/contenidos/programas/endireh/2021/doc/endireh2021_fd.pdf>

El análisis no parte de los microdatos originales, repartidos en 28 tablas, sino
del **archivo consolidado** que proporcionó la ayudantía del curso: un único CSV
con 110,127 registros y 24 columnas ya unidas y renombradas.

Ese archivo **no forma parte del repositorio**. El `.gitignore` excluye el
contenido de `data/data-raw/` por su tamaño, conforme a lo indicado en la
práctica; el archivo se entrega por separado. Para reproducir el
preprocesamiento hay que colocarlo en `data/data-raw/` antes de ejecutarlo. El
script lo localiza por su extensión, de modo que el nombre del archivo es
indistinto.

El resultado del preprocesamiento sí está en el repositorio, en
`data/data-processed/endireh_limpio.parquet` y su equivalente en CSV. Las
medidas, las figuras y los notebooks parten de ese Parquet, por lo que se pueden
ejecutar sin el archivo crudo. Es el archivo que la Práctica 4 llama
`endireh_2021_limpio.csv`.

## 3. Herramientas

El pipeline usa **polars** en todas sus etapas. Entre pandas y polars se eligió
polars siguiendo la recomendación de la práctica. A la escala de este archivo
la decisión no depende del rendimiento, sino de que polars obliga a escribir
las transformaciones como expresiones encadenadas y explícitas.

**numpy** calcula lo que polars no trae de forma nativa: las medidas ponderadas
por el factor de expansión (media, cuantiles y moda) y los índices de
heterogeneidad y concentración. También da a matplotlib las posiciones y los
arreglos con que dibuja las figuras.

**scipy** se usa para contrastar la entropía de Shannon del proyecto con
`scipy.stats.entropy`. **matplotlib** genera las figuras.

## 4. Requisitos e instalación

Requiere **Python 3.12 o superior**, que es el mínimo que admiten las versiones
fijadas de numpy y scipy; el proyecto se desarrolló con Python 3.14.

> [!NOTE]
> Las instrucciones son de acuerdo con el sistema que usamos, que en este caso es
> Arch Linux. En otros sistemas operativos, la instalación de Python y la creación
> del entorno virtual pueden variar.

```bash
# 1. Clonar el repositorio
git clone https://github.com/alangm-98/proyecto-endireh-violencia.git
cd proyecto-endireh-violencia

# 2. Crear y activar el entorno virtual
python -m venv env
source env/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt
```

El entorno debe llamarse `env`, `.venv` o `venv`: son los tres nombres que el
`.gitignore` excluye.

En Arch Linux, `python -m venv` requiere el paquete `python-pip` instalado a
nivel de sistema.

Dependencias fijadas en `requirements.txt`:

| Paquete      | Versión | Uso                                                |
| ------------ | ------- | -------------------------------------------------- |
| `polars`     | 1.44.2  | Framework de análisis del pipeline                 |
| `pyarrow`    | 25.0.1  | Soporte de formato Parquet                         |
| `numpy`      | 2.5.3   | Medidas ponderadas e índices de `indices.py`       |
| `scipy`      | 1.18.1  | Contraste de la entropía con `scipy.stats.entropy` |
| `matplotlib` | 3.11.2  | Generación de las figuras                          |
| `jupyterlab` | 4.6.3   | Ejecución de los notebooks                         |

## 5. Arquitectura del proyecto

```
proyecto-endireh-violencia/
├── config/
│   └── rutas.py                          # Rutas estáticas del proyecto
├── data/
│   ├── data-raw/                         # Archivo consolidado (excluido del repositorio)
│   ├── data-processed/                   # Datos limpios: endireh_limpio.parquet y .csv
│   ├── data-input-model/                 # Datos listos para modelado (vacía en esta etapa)
│   └── data-model/                       # Salidas del modelo (vacía en esta etapa)
├── notebooks/
│   ├── 01_medidas_localizacion.ipynb     # Media, mediana, moda y cuantiles
│   ├── 02_medidas_variabilidad.ipynb     # Rango, varianza, desviación, CV e IQR
│   ├── 03_medidas_heterogeneidad.ipynb   # Gini-Simpson, IQV y entropía de Shannon
│   ├── 04_medidas_concentracion.ipynb    # Curva de Lorenz y coeficiente de Gini
│   └── 05_gini_vs_entropia.ipynb         # Gini y entropía sobre una misma variable
├── reports/
│   └── figuras/                          # Las diez figuras en PNG
├── src/
│   ├── cleaning/
│   │   └── limpieza.py                   # Pipeline de preprocesamiento
│   ├── visualization/
│   │   ├── eda.py                        # Medidas de localización y variabilidad
│   │   ├── indices.py                    # Índices de heterogeneidad y concentración
│   │   └── graficas.py                   # Construcción de las figuras
│   └── models/                           # Scripts de modelado (vacía en esta etapa)
├── .gitignore
├── README.md
└── requirements.txt
```

Las carpetas `data-input-model/`, `data-model/` y `src/models/` corresponden a
la etapa de modelado, y permanecen vacías: la estructura de directorios de la
práctica las incluye. Cada una contiene un archivo `.gitkeep` para que Git
conserve la carpeta vacía.

Ningún script escribe rutas a mano: todas se derivan de `config/rutas.py`, que
resuelve la raíz del proyecto a partir de la ubicación del propio archivo.

## 6. Cómo ejecutar el pipeline

Con el entorno activo y desde la raíz del proyecto.

```bash
# 1. Preprocesamiento
python src/cleaning/limpieza.py
```

Requiere el archivo consolidado en `data/data-raw/`. Lee el CSV crudo, repara
el encoding, convierte los códigos de no respuesta a nulo, ajusta los tipos,
elimina duplicados, imputa donde procede y escribe
`data/data-processed/endireh_limpio.parquet` y su equivalente en CSV. Imprime un
registro de cuántas filas y valores modificó en cada paso. Como el resultado ya
está en el repositorio, este paso solo hace falta para reproducirlo.

```bash
# 2. Medidas descriptivas
python src/visualization/eda.py
```

Imprime las frecuencias de las variables cualitativas, las medidas de
localización en versión simple y ponderada, las de variabilidad comparando el
grupo que reportó violencia de pareja contra el que no, y la prevalencia
ponderada.

```bash
# 3. Comprobación de los índices
python src/visualization/indices.py
```

Comprueba las funciones de Gini-Simpson, IQV, entropía de Shannon, curva de
Lorenz y coeficiente de Gini contra casos de valor conocido, y contrasta la
entropía con `scipy.stats.entropy`.

```bash
# 4. Figuras
python src/visualization/graficas.py
```

Genera las diez figuras en `reports/figuras/`.

```bash
# 5. Notebooks
jupyter lab notebooks/
```

Hay un notebook por tipo de análisis. Cada uno abre con la teoría de sus
medidas y presenta los resultados con su interpretación.

| Notebook                          | Contenido                                                            |
| --------------------------------- | -------------------------------------------------------------------- |
| `01_medidas_localizacion.ipynb`   | Medidas de localización, simples y ponderadas                        |
| `02_medidas_variabilidad.ipynb`   | Medidas de variabilidad por grupo                                    |
| `03_medidas_heterogeneidad.ipynb` | Gini-Simpson, IQV y entropía de Shannon sobre variables cualitativas |
| `04_medidas_concentracion.ipynb`  | Curva de Lorenz y coeficiente de Gini de los casos por entidad       |
| `05_gini_vs_entropia.ipynb`       | Comparación de Gini y entropía sobre el estado conyugal              |

Los notebooks importan de los scripts de `src/` y no recalculan nada, de modo
que cada resultado tiene un único origen. Solo necesitan el Parquet procesado:
cada uno genera las figuras que muestra.

## 7. Salida del preprocesamiento

|                        | Archivo crudo | Procesado  |
| ---------------------- | ------------- | ---------- |
| Registros              | 110,127       | 105,752    |
| Columnas               | 24            | 25         |
| Población representada | 50,523,469    | 49,144,897 |

La columna adicional es `ingreso_pareja_imputado`, que marca los registros cuyo
ingreso fue imputado.

Los reportes de las prácticas documentan el detalle de cada paso, las decisiones
tomadas y las limitaciones encontradas en tres variables del archivo
consolidado.

## 8. Equipo

- Flores Juárez Luis Enrique
- García Morales Carlos Alan
- Rangel Salcedo Melanie Valeria
