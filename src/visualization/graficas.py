"""
Gráficas del análisis exploratorio de la ENDIREH 2021.

Construye diez figuras y las guarda en reports/figuras/:

    01_distribucion_edad_primer_union.png
    02_ingreso_por_violencia.png
    03_prevalencia_por_entidad.png
    04_prevalencia_por_estado_civil.png
    05_escolaridad_muestra_vs_poblacion.png
    06_distribucion_ingreso_pareja.png
    07_boxplot_ingreso_por_violencia.png
    08_heterogeneidad_iqv.png
    09_curva_lorenz_entidades.png
    10_estado_conyugal_poblacion_y_casos.png

Uso:
    python3 src/visualization/graficas.py
"""

import sys
from pathlib import Path

import matplotlib

# Agg dibuja en memoria; permite generar las figuras sin entorno gráfico.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import BASE_DIR, RUTA_ENDIREH_LIMPIO, RUTA_FIGURAS
from src.visualization.eda import medidas_localizacion
from src.visualization.indices import (
    curva_lorenz,
    entropia_shannon,
    gini,
    tabla_entidades,
    tabla_heterogeneidad,
    tabla_sintesis,
)

# Paleta común a todas las figuras.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
AZUL = "#2a78d6"  # categoría 1 / serie única
NARANJA = "#eb6834"  # categoría 2
GRIS_RETICULA = "#d8d7d2"

# Nombre con que cada variable aparece en las figuras.
NOMBRES_VARIABLES = {
    "estado_civil_desc": "Estado conyugal",
    "estrato_socioeconomico": "Estrato socioeconómico",
    "nom_entidad": "Entidad federativa",
    "sufrio_violencia_pareja": "Violencia de pareja reportada",
}

# Valores por omisión de matplotlib para todo el módulo.
plt.rcParams.update(
    {
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SUAVE,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
        "axes.edgecolor": GRIS_RETICULA,
        "font.size": 10,
        "figure.dpi": 130,
    }
)


def preparar_ejes(ax, eje_valor: str = "y") -> None:
    """
    Deja los ejes con retícula tenue y sin marcos superfluos.

    eje_valor indica cuál de los dos ejes lleva la magnitud y recibe retícula.
    """
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(GRIS_RETICULA)
    ax.spines["bottom"].set_color(GRIS_RETICULA)
    ax.grid(axis=eje_valor, color=GRIS_RETICULA, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def titular(ax, titulo: str, subtitulo: str) -> None:
    """Escribe el título y el subtítulo por encima del área de graficado."""
    ax.set_title(
        titulo, fontsize=13, fontweight="bold", color=TINTA, loc="left", pad=26
    )
    ax.annotate(
        subtitulo,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9.5,
        color=TINTA_SUAVE,
        va="bottom",
        ha="left",
    )


def marcar_nacional(ax, valor: float) -> None:
    """Dibuja la línea de referencia con la prevalencia nacional."""
    ax.axvline(valor, color=NARANJA, linewidth=1.6, linestyle="--")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.07)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        f"Nacional {valor:.1f}%",
        xy=(valor, 1),
        xycoords=trans,
        xytext=(5, -10),
        textcoords="offset points",
        color=NARANJA,
        fontsize=9,
        va="top",
        ha="left",
    )


def pie_de_figura(ax, texto: str) -> None:
    """Escribe la nota al pie con la fuente de los datos y el tamaño de muestra."""
    ax.annotate(
        texto,
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -52),
        textcoords="offset points",
        fontsize=8,
        color=TINTA_SUAVE,
        va="top",
        ha="left",
    )


def guardar(fig, nombre: str) -> None:
    """Escribe la figura en reports/figuras/ y libera la memoria."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficas] {ruta.relative_to(BASE_DIR)}")


def prevalencia_ponderada(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    """
    Calcula la prevalencia de violencia dentro de cada categoría de la columna.

    Divide la suma de factores de expansión de las mujeres que reportaron
    violencia entre la suma de factores de todo el grupo.
    """
    return (
        df.group_by(columna)
        .agg(
            (pl.col("sufrio_violencia_pareja") * pl.col("factor_expansion"))
            .sum()
            .alias("casos"),
            pl.col("factor_expansion").sum().alias("total"),
            pl.len().alias("n"),
        )
        .with_columns((100 * pl.col("casos") / pl.col("total")).alias("prevalencia"))
        .sort("prevalencia")
    )


# --------------------------------------------------------------------------


def figura_01_distribucion_edad(df: pl.DataFrame) -> None:
    """
    Histograma de edad_primer_union.

    Sombrea el tramo de valores menores a 10 y lo anota con su porcentaje.
    """
    valores = df["edad_primer_union"].drop_nulls()
    imposibles = (valores < 10).sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    # Un bin por cada valor entero.
    ax.hist(
        valores.to_numpy(),
        bins=np.arange(0, valores.max() + 2, 1),
        color=AZUL,
        edgecolor=SUPERFICIE,
        linewidth=0.4,
    )

    ax.axvspan(-0.5, 9.5, color=NARANJA, alpha=0.10, zorder=0)
    ax.axvline(9.5, color=NARANJA, linewidth=1.6, linestyle="--")
    altura = ax.get_ylim()[1] * 0.72
    ax.annotate(
        f"{imposibles:,} valores ({100 * imposibles / len(valores):.1f}%)\n"
        "por debajo de 10 años",
        xy=(9.5, altura),
        xytext=(24, altura),
        fontsize=9,
        color=NARANJA,
        va="center",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=NARANJA, linewidth=1.2),
    )

    preparar_ejes(ax)
    titular(
        ax,
        "Distribución de la variable edad_primer_union",
        "Declarada como edad a la primera unión, pero el "
        f"{100 * imposibles / len(valores):.1f}% de los valores es menor a 10",
    )
    ax.set_xlabel("Valor declarado")
    ax.set_ylabel("Mujeres encuestadas")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(valores):,} registros con valor.\n"
        "Los códigos 98 y 99 se excluyen por corresponder a no respuesta.",
    )
    guardar(fig, "01_distribucion_edad_primer_union.png")


def figura_02_ingreso_por_violencia(df: pl.DataFrame) -> None:
    """
    Compara el ingreso de la pareja entre los dos grupos.

    Enfrenta ambos grupos en los percentiles 10, 25, 50, 75 y 90.
    """
    sub = df.filter(
        pl.col("ingreso_pareja").is_not_null() & (pl.col("ingreso_pareja") > 0)
    )
    con = sub.filter(pl.col("sufrio_violencia_pareja") == 1)["ingreso_pareja"]
    sin = sub.filter(pl.col("sufrio_violencia_pareja") == 0)["ingreso_pareja"]

    etiquetas = ["P10", "Q1 (P25)", "Mediana", "Q3 (P75)", "P90"]
    cortes = [0.10, 0.25, 0.50, 0.75, 0.90]
    v_con = [con.quantile(q, interpolation="linear") for q in cortes]
    v_sin = [sin.quantile(q, interpolation="linear") for q in cortes]

    y = np.arange(len(etiquetas))
    alto = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + alto / 2, v_sin, height=alto, color=AZUL, label="No reportó violencia")
    ax.barh(y - alto / 2, v_con, height=alto, color=NARANJA, label="Reportó violencia")

    for yi, (vs, vc) in enumerate(zip(v_sin, v_con)):
        ax.text(
            vs + 150,
            yi + alto / 2,
            f"${vs:,.0f}",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )
        ax.text(
            vc + 150,
            yi - alto / 2,
            f"${vc:,.0f}",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )

    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, max(v_sin + v_con) * 1.18)
    preparar_ejes(ax, eje_valor="x")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    titular(
        ax,
        "La diferencia de ingreso se concentra en la mitad alta de la distribución",
        "Ingreso mensual de la pareja por percentil, según si se reportó violencia",
    )
    ax.set_xlabel("Ingreso de la pareja (pesos)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(con):,} con violencia y {len(sin):,} sin violencia; "
        "se excluyen los ingresos en cero.\n"
        "Hasta la mediana ambas distribuciones coinciden; la diferencia aparece en Q3 y P90.\n"
        "La asociación no permite establecer causalidad: también puede reflejar diferencias "
        "en la disposición a reportar.",
    )
    guardar(fig, "02_ingreso_por_violencia.png")


def figura_03_prevalencia_entidad(df: pl.DataFrame) -> None:
    resumen = prevalencia_ponderada(df, "nom_entidad")
    nombres = resumen["nom_entidad"].to_list()
    valores = resumen["prevalencia"].to_numpy()
    nacional = 100 * (
        (df["sufrio_violencia_pareja"] * df["factor_expansion"]).sum()
        / df["factor_expansion"].sum()
    )

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(nombres, valores, color=AZUL, height=0.72)
    marcar_nacional(ax, nacional)

    for y, v in enumerate(valores):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8.5, color=TINTA_SUAVE)

    ax.set_xlim(0, valores.max() * 1.15)
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Prevalencia de violencia de pareja por entidad federativa",
        "Porcentaje ponderado de mujeres de 15 años y más que la reportaron",
    )
    ax.set_xlabel("Prevalencia (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros ponderados por factor_expansion.\n"
        "La cifra corresponde a la violencia reportada en la encuesta, no a su ocurrencia.",
    )
    guardar(fig, "03_prevalencia_por_entidad.png")


def figura_04_prevalencia_estado_civil(df: pl.DataFrame) -> None:
    resumen = prevalencia_ponderada(df, "estado_civil_desc")
    nombres = [str(x) for x in resumen["estado_civil_desc"].to_list()]
    valores = resumen["prevalencia"].to_numpy()
    nacional = 100 * (
        (df["sufrio_violencia_pareja"] * df["factor_expansion"]).sum()
        / df["factor_expansion"].sum()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(nombres, valores, color=AZUL, height=0.62)
    marcar_nacional(ax, nacional)

    for y, v in enumerate(valores):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=9.5, color=TINTA_SUAVE)

    ax.set_xlim(0, valores.max() * 1.18)
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Las mujeres separadas y divorciadas reportan el triple que las casadas",
        "Prevalencia ponderada de violencia de pareja por estado conyugal",
    )
    ax.set_xlabel("Prevalencia (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros ponderados.\n"
        "La encuesta es transversal: registra el estado conyugal y la violencia en un mismo\n"
        "momento, por lo que no permite establecer el orden entre ambos.",
    )
    guardar(fig, "04_prevalencia_por_estado_civil.png")


def figura_05_escolaridad(df: pl.DataFrame) -> None:
    """
    Distribución de nivel_escolaridad en la muestra y en la población.

    Enfrenta el porcentaje de cada categoría antes y después de aplicar el
    factor de expansión.
    """
    resumen = (
        df.group_by("nivel_escolaridad")
        .agg(
            pl.len().alias("n"),
            pl.col("factor_expansion").sum().alias("poblacion"),
        )
        .with_columns(
            (100 * pl.col("n") / df.height).alias("pct_muestra"),
            (100 * pl.col("poblacion") / df["factor_expansion"].sum()).alias("pct_pob"),
        )
        # Orden natural de las categorías, no por frecuencia.
        .sort("nivel_escolaridad")
    )
    categorias = [str(x) for x in resumen["nivel_escolaridad"].to_list()]
    muestra = resumen["pct_muestra"].to_numpy()
    poblacion = resumen["pct_pob"].to_numpy()

    y = np.arange(len(categorias))
    alto = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(
        y + alto / 2, poblacion, height=alto, color=AZUL, label="Población estimada"
    )
    ax.barh(
        y - alto / 2, muestra, height=alto, color=NARANJA, label="Muestra encuestada"
    )

    for yi, (pob, mue) in enumerate(zip(poblacion, muestra)):
        ax.text(
            pob + 0.6,
            yi + alto / 2,
            f"{pob:.1f}%",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )
        ax.text(
            mue + 0.6,
            yi - alto / 2,
            f"{mue:.1f}%",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )

    ax.set_yticks(y, categorias)
    ax.invert_yaxis()
    ax.set_xlim(0, max(poblacion.max(), muestra.max()) * 1.20)
    preparar_ejes(ax, eje_valor="x")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    titular(
        ax,
        "La ponderación redistribuye el peso entre categorías",
        "Distribución de nivel_escolaridad en la muestra y en la población estimada",
    )
    ax.set_xlabel("Porcentaje del total (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros; la población estimada pondera "
        "cada registro por factor_expansion.\n"
        "A1 pierde 7.4 puntos al ponderar y C1 gana 5.3. Esa redistribución es la razón por "
        "la que una medida simple y\nsu versión ponderada no coinciden. Las categorías A1 a "
        "C2 no corresponden al catálogo educativo del INEGI\ny se presentan sin interpretar "
        "su contenido.",
    )
    guardar(fig, "05_escolaridad_muestra_vs_poblacion.png")


def figura_06_distribucion_ingreso(df: pl.DataFrame) -> None:
    """
    Histograma ponderado de ingreso_pareja.

    Cada barra suma el factor de expansión de un intervalo de 500 pesos. Marca
    la media y la mediana ponderadas y corta el eje en 20,000 pesos.
    """
    sub = df.filter(pl.col("ingreso_pareja").is_not_null())
    m = medidas_localizacion(sub["ingreso_pareja"], sub["factor_expansion"])
    tope, ancho = 20_000, 500
    visible = sub.filter(pl.col("ingreso_pareja") <= tope)
    pct_fuera = 100 * (1 - visible["factor_expansion"].sum() / m["poblacion"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        visible["ingreso_pareja"].to_numpy(),
        bins=np.arange(0, tope + ancho, ancho),
        weights=visible["factor_expansion"].to_numpy(),
        color=AZUL,
        edgecolor=SUPERFICIE,
        linewidth=0.4,
    )
    ax.axvline(
        m["mediana_pond"],
        color=NARANJA,
        linewidth=1.8,
        label=f"Mediana ponderada: ${m['mediana_pond']:,.0f}",
    )
    ax.axvline(
        m["media_pond"],
        color=NARANJA,
        linewidth=1.8,
        linestyle="--",
        label=f"Media ponderada: ${m['media_pond']:,.0f}",
    )

    ax.set_xlim(0, tope)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}"))
    preparar_ejes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    titular(
        ax,
        "Distribución del ingreso de la pareja",
        "Población estimada por intervalo de 500 pesos, con su media y su mediana",
    )
    ax.set_xlabel("Ingreso de la pareja (pesos)")
    ax.set_ylabel("Mujeres (millones)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {m['n']:,} registros con ingreso de la pareja, "
        f"que representan a {m['poblacion']:,.0f} mujeres;\n"
        f"incluye valores imputados. El eje se corta en {tope:,} pesos: queda fuera "
        f"el {pct_fuera:.1f}% de esa población,\n"
        f"con ingresos de hasta {sub['ingreso_pareja'].max():,.0f} pesos.",
    )
    guardar(fig, "06_distribucion_ingreso_pareja.png")


def resumen_caja(valores: pl.Series) -> dict:
    """
    Devuelve los valores con que se dibuja un diagrama de caja.

    Los cuartiles usan interpolación lineal. Los bigotes llegan al último valor
    que queda dentro de 1.5 veces el rango intercuartílico contado desde la
    caja; n_atipicos cuenta los valores que quedan más allá.
    """
    q1, med, q3 = (
        valores.quantile(q, interpolation="linear") for q in (0.25, 0.50, 0.75)
    )
    margen = 1.5 * (q3 - q1)
    dentro = valores.filter((valores >= q1 - margen) & (valores <= q3 + margen))
    return {
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": dentro.min(),
        "whishi": dentro.max(),
        "n": valores.len(),
        "n_atipicos": valores.len() - dentro.len(),
    }


def figura_07_boxplot_ingreso(df: pl.DataFrame) -> None:
    """
    Diagrama de caja de ingreso_pareja para los dos grupos.

    Usa cuartiles sin ponderar, escala lineal y no dibuja los valores atípicos.
    """
    sub = df.filter(pl.col("ingreso_pareja").is_not_null())
    sin = resumen_caja(
        sub.filter(pl.col("sufrio_violencia_pareja") == 0)["ingreso_pareja"]
    )
    con = resumen_caja(
        sub.filter(pl.col("sufrio_violencia_pareja") == 1)["ingreso_pareja"]
    )
    grupos = [
        ("Reportó violencia", con, NARANJA),
        ("No reportó violencia", sin, AZUL),
    ]

    claves = ("q1", "med", "q3", "whislo", "whishi")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    cajas = ax.bxp(
        [{clave: g[clave] for clave in claves} for _, g, _ in grupos],
        positions=range(len(grupos)),
        orientation="horizontal",
        widths=0.46,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=TINTA, linewidth=2),
        whiskerprops=dict(color=TINTA_SUAVE, linewidth=1.2),
        capprops=dict(color=TINTA_SUAVE, linewidth=1.2),
        boxprops=dict(edgecolor=SUPERFICIE, linewidth=0.8),
    )
    for caja, (_, _, color) in zip(cajas["boxes"], grupos):
        caja.set_facecolor(color)
    for y, (_, g, _) in enumerate(grupos):
        ax.text(
            g["q1"],
            y + 0.33,
            f"Q1 ${g['q1']:,.0f}  ·  Mediana ${g['med']:,.0f}  ·  Q3 ${g['q3']:,.0f}",
            va="bottom",
            fontsize=9,
            color=TINTA_SUAVE,
        )

    limite = max(g["whishi"] for _, g, _ in grupos)
    ax.set_xlim(-0.02 * limite, 1.05 * limite)
    ax.set_ylim(-0.6, len(grupos) - 0.2)
    ax.set_yticks(range(len(grupos)), [nombre for nombre, _, _ in grupos])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Ingreso de la pareja según si se reportó violencia",
        "Diagrama de caja con cuartiles sin ponderar; no se dibujan los valores atípicos",
    )
    ax.set_xlabel("Ingreso de la pareja (pesos)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {con['n']:,} con violencia y {sin['n']:,} sin "
        "violencia; incluye ingresos en cero y valores imputados.\n"
        "Los bigotes llegan al último valor dentro de 1.5 veces el rango "
        "intercuartílico.\n"
        f"No se dibujan los valores atípicos: {con['n_atipicos']:,} con violencia "
        f"({100 * con['n_atipicos'] / con['n']:.1f}%) y {sin['n_atipicos']:,} sin "
        f"violencia ({100 * sin['n_atipicos'] / sin['n']:.1f}%).\n"
        "La diferencia entre grupos describe lo reportado en la encuesta y no permite "
        "establecer causalidad.",
    )
    guardar(fig, "07_boxplot_ingreso_por_violencia.png")


def figura_08_heterogeneidad(df: pl.DataFrame) -> None:
    """
    Barras con el IQV ponderado de las variables cualitativas.

    Ordena las variables de mayor a menor IQV y anota junto a cada una su
    número de categorías.
    """
    tabla = tabla_heterogeneidad(df).sort("iqv")
    etiquetas = [
        f"{NOMBRES_VARIABLES[fila['variable']]}\n{fila['k']} categorías"
        for fila in tabla.iter_rows(named=True)
    ]
    valores = tabla["iqv"].to_list()

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(etiquetas, valores, color=AZUL, height=0.6)
    for y, v in enumerate(valores):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=9.5, color=TINTA_SUAVE)

    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Heterogeneidad de las variables cualitativas",
        "Índice de variación cualitativa (IQV): 0 si una categoría reúne todo, "
        "1 si todas pesan igual",
    )
    ax.set_xlabel("IQV")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros; la proporción de cada "
        "categoría se pondera por factor_expansion.\n"
        "El IQV divide el índice de Gini-Simpson entre su máximo, (k-1)/k, lo que "
        "permite comparar variables\ncon distinto número de categorías.",
    )
    guardar(fig, "08_heterogeneidad_iqv.png")


def figura_09_curva_lorenz(df: pl.DataFrame) -> None:
    """
    Curva de Lorenz de los casos ponderados por entidad.

    Dibuja como referencia la curva de la población y la recta de reparto
    igual. Cada curva ordena las entidades según su propia variable.
    """
    tabla = tabla_entidades(df)
    curvas = [
        (
            "Casos de violencia de pareja reportada",
            tabla["casos_ponderados"],
            AZUL,
            "-",
        ),
        ("Población de 15 años y más", tabla["poblacion_ponderada"], NARANJA, "--"),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    ax.plot(
        [0, 1],
        [0, 1],
        color=TINTA_SUAVE,
        linewidth=1,
        linestyle=":",
        label="Reparto igual entre entidades",
    )
    for nombre, valores, color, estilo in curvas:
        unidades, total = curva_lorenz(valores)
        ax.plot(
            unidades,
            total,
            color=color,
            linewidth=2,
            linestyle=estilo,
            label=f"{nombre} (Gini {gini(valores):.3f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    preparar_ejes(ax)
    ax.grid(axis="x", color=GRIS_RETICULA, linewidth=0.6, alpha=0.9)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    titular(
        ax,
        "Concentración de los casos de violencia por entidad",
        "Curva de Lorenz de los casos ponderados y, como referencia, de la población",
    )
    ax.set_xlabel("Entidades acumuladas, de menor a mayor")
    ax.set_ylabel("Parte acumulada del total")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). {tabla.height} entidades; casos y población "
        "ponderados por factor_expansion.\n"
        "Cada curva ordena las entidades de menor a mayor según su propia variable. "
        f"El Gini máximo\ncon {tabla.height} entidades es "
        f"{(tabla.height - 1) / tabla.height:.3f}. Los casos corresponden a la "
        "violencia reportada en la encuesta,\nno a su ocurrencia.",
    )
    guardar(fig, "09_curva_lorenz_entidades.png")


def figura_10_sintesis_estado_conyugal(df: pl.DataFrame) -> None:
    """
    Barras con la parte de la población y de los casos en cada estado conyugal.

    Marca el reparto igual entre categorías y anota en la leyenda la entropía y
    el Gini de cada distribución.
    """
    tabla = tabla_sintesis(df)
    categorias = [str(x) for x in tabla["estado_civil_desc"].to_list()]
    k = len(categorias)
    series = [
        ("Casos reportados", tabla["casos"], NARANJA, -1),
        ("Población", tabla["poblacion"], AZUL, 1),
    ]

    y = np.arange(k)
    alto = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for nombre, valores, color, lado in series:
        partes = (100 * valores / valores.sum()).to_list()
        ax.barh(
            y + lado * alto / 2,
            partes,
            height=alto,
            color=color,
            label=f"{nombre}: entropía {entropia_shannon(valores):.2f}, "
            f"Gini {gini(valores):.3f}",
        )
        for yi, parte in zip(y, partes):
            ax.text(
                parte + 0.5,
                yi + lado * alto / 2,
                f"{parte:.1f}%",
                va="center",
                fontsize=9,
                color=TINTA_SUAVE,
                bbox=dict(facecolor=SUPERFICIE, edgecolor="none", pad=1.5),
            )
    ax.axvline(
        100 / k,
        color=TINTA_SUAVE,
        linewidth=1.2,
        linestyle=":",
        zorder=0,
        label=f"Reparto igual: {100 / k:.1f}%",
    )

    ax.set_yticks(y, categorias)
    ax.set_xlim(0, 42)
    preparar_ejes(ax, eje_valor="x")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1], frameon=False, loc="lower right", fontsize=9.5
    )
    titular(
        ax,
        "Población y casos de violencia de pareja por estado conyugal",
        "Parte del total que corresponde a cada categoría, frente al reparto igual",
    )
    ax.set_xlabel("Porcentaje del total (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros ponderados por "
        "factor_expansion; los casos son la violencia reportada\nen la encuesta. "
        f"Entropía en bits. Máximos con {k} categorías: entropía de "
        f"{np.log2(k):.2f} y Gini de {(k - 1) / k:.3f}.",
    )
    guardar(fig, "10_estado_conyugal_poblacion_y_casos.png")


# --------------------------------------------------------------------------


def generar_graficas() -> None:
    if not RUTA_ENDIREH_LIMPIO.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_ENDIREH_LIMPIO}. "
            "Ejecuta antes: python3 src/cleaning/limpieza.py"
        )
    df = pl.read_parquet(RUTA_ENDIREH_LIMPIO)
    print(f"[graficas] Dataset: {df.height:,} filas. Guardando en {RUTA_FIGURAS}")

    figura_01_distribucion_edad(df)
    figura_02_ingreso_por_violencia(df)
    figura_03_prevalencia_entidad(df)
    figura_04_prevalencia_estado_civil(df)
    figura_05_escolaridad(df)
    figura_06_distribucion_ingreso(df)
    figura_07_boxplot_ingreso(df)
    figura_08_heterogeneidad(df)
    figura_09_curva_lorenz(df)
    figura_10_sintesis_estado_conyugal(df)
    print("[graficas] Listo: 10 figuras generadas.")


if __name__ == "__main__":
    generar_graficas()
