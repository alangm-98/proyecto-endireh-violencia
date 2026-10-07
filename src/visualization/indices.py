"""
Índices de heterogeneidad y de concentración.

Reúne las funciones que usan los notebooks de heterogeneidad, concentración y
síntesis: Gini-Simpson, IQV y entropía de Shannon para variables cualitativas,
y curva de Lorenz y coeficiente de Gini para el reparto de una cantidad entre
unidades.

Los índices reciben las frecuencias ya agregadas, una por categoría o por
unidad. Pueden ser conteos de la muestra o sumas de factor_expansion.

Ejecutado como script comprueba las funciones contra casos de valor conocido y
contrasta la entropía con scipy.stats.entropy.

Uso:
    python3 src/visualization/indices.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import entropy

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_ENDIREH_LIMPIO
from src.visualization.eda import titulo

# Variables cualitativas sobre las que se miden los índices de heterogeneidad.
VARIABLES_HETEROGENEIDAD = [
    "estado_civil_desc",
    "estrato_socioeconomico",
    "nom_entidad",
    "sufrio_violencia_pareja",
]

TOLERANCIA = 1e-12


# --------------------------------------------------------------------------
# Frecuencias
# --------------------------------------------------------------------------


def tabla_frecuencias(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    """
    Devuelve una fila por categoría con su frecuencia y su población.

    La frecuencia cuenta los registros de la muestra y la población suma su
    factor_expansion. Los registros en que la columna es nula se excluyen.
    """
    return (
        df.filter(pl.col(columna).is_not_null())
        .group_by(columna)
        .agg(
            pl.len().alias("frecuencia"),
            pl.col("factor_expansion").sum().alias("poblacion"),
        )
        .sort(columna)
    )


def validar(frecuencias) -> np.ndarray:
    """
    Devuelve las frecuencias como arreglo de flotantes.

    Lanza ValueError si no forman una lista de valores finitos y no negativos
    con suma mayor a cero.
    """
    f = np.asarray(frecuencias, dtype=float)
    if f.ndim != 1 or f.size == 0:
        raise ValueError("Se esperaba una lista de frecuencias no vacía.")
    if not np.isfinite(f).all() or (f < 0).any() or f.sum() == 0:
        raise ValueError(
            "Las frecuencias deben ser finitas, no negativas y sumar más de cero."
        )
    return f


def proporciones(frecuencias) -> np.ndarray:
    """Devuelve la proporción del total que corresponde a cada frecuencia."""
    f = validar(frecuencias)
    return f / f.sum()


# --------------------------------------------------------------------------
# Heterogeneidad
# --------------------------------------------------------------------------


def gini_simpson(frecuencias) -> float:
    """
    Devuelve el índice de Gini-Simpson: 1 menos la suma de p al cuadrado.

    Vale 0 cuando una categoría reúne todos los casos y (k-1)/k cuando las k
    categorías tienen la misma proporción.
    """
    p = proporciones(frecuencias)
    return float(1 - np.sum(p**2))


def iqv(frecuencias) -> float:
    """
    Devuelve el índice de variación cualitativa: k(1 - suma de p²)/(k-1).

    Es el Gini-Simpson dividido entre su máximo, de modo que va de 0 a 1 con
    cualquier número de categorías. Lanza ValueError si hay menos de dos.
    """
    p = proporciones(frecuencias)
    k = len(p)
    if k < 2:
        raise ValueError("El IQV necesita al menos dos categorías.")
    return float(k * (1 - np.sum(p**2)) / (k - 1))


def entropia_shannon(frecuencias) -> float:
    """
    Devuelve la entropía de Shannon en bits: la suma de p·log2(1/p).

    Las categorías con frecuencia cero no aportan. Vale 0 cuando una categoría
    reúne todos los casos y log2(k) cuando las k tienen la misma proporción.
    """
    p = proporciones(frecuencias)
    p = p[p > 0]
    return float(np.sum(p * np.log2(1 / p)))


def entropia_normalizada(frecuencias) -> float:
    """
    Devuelve la entropía de Shannon dividida entre su máximo, log2(k).

    Va de 0 a 1 con cualquier número de categorías. Lanza ValueError si hay
    menos de dos.
    """
    k = len(validar(frecuencias))
    if k < 2:
        raise ValueError("La entropía normalizada necesita al menos dos categorías.")
    return entropia_shannon(frecuencias) / float(np.log2(k))


def tabla_heterogeneidad(
    df: pl.DataFrame,
    columnas: list[str] = VARIABLES_HETEROGENEIDAD,
    ponderada: bool = True,
) -> pl.DataFrame:
    """
    Devuelve una fila por variable con sus índices de heterogeneidad.

    Con ponderada=True las proporciones salen de la población de cada
    categoría; con False, de su frecuencia en la muestra. entropia_maxima es
    log2(k).
    """
    peso = "poblacion" if ponderada else "frecuencia"
    filas = []
    for columna in columnas:
        frecuencias = tabla_frecuencias(df, columna)[peso]
        filas.append(
            {
                "variable": columna,
                "k": frecuencias.len(),
                "gini_simpson": gini_simpson(frecuencias),
                "iqv": iqv(frecuencias),
                "entropia": entropia_shannon(frecuencias),
                "entropia_maxima": float(np.log2(frecuencias.len())),
                "entropia_normalizada": entropia_normalizada(frecuencias),
            }
        )
    return pl.DataFrame(filas)


def seccion_heterogeneidad(
    df: pl.DataFrame,
    columnas: list[str] = VARIABLES_HETEROGENEIDAD,
    ponderada: bool = True,
) -> None:
    """Imprime la tabla de índices de heterogeneidad de las columnas indicadas."""
    base = "población estimada" if ponderada else "muestra sin ponderar"
    titulo(f"MEDIDAS DE HETEROGENEIDAD ({base})")
    print(
        f"\n    {'Variable':<26}{'k':>3}{'Gini-Simpson':>15}{'IQV':>9}"
        f"{'Entropía':>11}{'Máxima':>9}{'Normalizada':>14}"
    )
    print(f"    {'-' * 87}")
    for fila in tabla_heterogeneidad(df, columnas, ponderada).iter_rows(named=True):
        print(
            f"    {fila['variable']:<26}{fila['k']:>3}{fila['gini_simpson']:>15.4f}"
            f"{fila['iqv']:>9.4f}{fila['entropia']:>11.4f}"
            f"{fila['entropia_maxima']:>9.4f}{fila['entropia_normalizada']:>14.4f}"
        )


# --------------------------------------------------------------------------
# Concentración
# --------------------------------------------------------------------------


def curva_lorenz(valores) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve los puntos de la curva de Lorenz de los valores.

    Ordena las unidades de menor a mayor y devuelve dos arreglos de n+1 puntos
    que empiezan en 0 y terminan en 1: la fracción acumulada de unidades y la
    fracción acumulada del total que reúnen.
    """
    v = np.sort(validar(valores))
    acumulado = np.cumsum(v)
    unidades = np.arange(len(v) + 1) / len(v)
    total = np.concatenate([[0.0], acumulado / acumulado[-1]])
    return unidades, total


def gini(valores) -> float:
    """
    Devuelve el coeficiente de Gini de los valores.

    Con los valores ordenados de menor a mayor calcula
    (2·Σ i·x_i - (n+1)·Σ x_i) / (n·Σ x_i). Vale 0 cuando todas las unidades
    tienen el mismo valor y como máximo (n-1)/n, cuando una sola reúne el total.
    """
    v = np.sort(validar(valores))
    n = len(v)
    posiciones = np.arange(1, n + 1)
    return float((2 * np.sum(posiciones * v) - (n + 1) * v.sum()) / (n * v.sum()))


def tabla_entidades(df: pl.DataFrame) -> pl.DataFrame:
    """
    Devuelve una fila por entidad con sus casos, su población y su prevalencia.

    casos_ponderados suma el factor_expansion de las mujeres que reportaron
    violencia de pareja y poblacion_ponderada el de todas. La prevalencia es su
    cociente, en porcentaje.
    """
    return (
        df.group_by("nom_entidad")
        .agg(
            pl.col("factor_expansion")
            .filter(pl.col("sufrio_violencia_pareja") == 1)
            .sum()
            .alias("casos_ponderados"),
            pl.col("factor_expansion").sum().alias("poblacion_ponderada"),
        )
        .with_columns(
            (100 * pl.col("casos_ponderados") / pl.col("poblacion_ponderada")).alias(
                "prevalencia"
            )
        )
        .sort("nom_entidad")
    )


def seccion_entidades(df: pl.DataFrame) -> None:
    """Imprime los casos, la población y la prevalencia de cada entidad."""
    tabla = tabla_entidades(df).sort("casos_ponderados", descending=True)
    casos = tabla["casos_ponderados"].sum()
    poblacion = tabla["poblacion_ponderada"].sum()

    titulo("CASOS DE VIOLENCIA DE PAREJA REPORTADA POR ENTIDAD (ponderados)")
    print(
        f"\n    {'Entidad':<33}{'Casos':>11}{'% casos':>10}"
        f"{'Población':>13}{'% pobl.':>10}{'Prevalencia':>14}"
    )
    print(f"    {'-' * 91}")
    for fila in tabla.iter_rows(named=True):
        print(
            f"    {fila['nom_entidad']:<33}{fila['casos_ponderados']:>11,.0f}"
            f"{100 * fila['casos_ponderados'] / casos:>9.1f}%"
            f"{fila['poblacion_ponderada']:>13,.0f}"
            f"{100 * fila['poblacion_ponderada'] / poblacion:>9.1f}%"
            f"{fila['prevalencia']:>13.1f}%"
        )
    print(f"    {'-' * 91}")
    print(
        f"    {'Total':<33}{casos:>11,.0f}{100:>9.1f}%{poblacion:>13,.0f}{100:>9.1f}%"
        f"{100 * casos / poblacion:>13.1f}%"
    )


def seccion_concentracion(df: pl.DataFrame) -> None:
    """
    Imprime el Gini de casos, población y prevalencia entre las entidades.

    Agrega dos lecturas de la curva de Lorenz: la parte del total que reúne la
    mitad de las entidades con valores más bajos y la que reúnen las cinco con
    valores más altos.
    """
    tabla = tabla_entidades(df)
    n = tabla.height
    series = [
        ("Casos ponderados", tabla["casos_ponderados"]),
        ("Población", tabla["poblacion_ponderada"]),
        ("Prevalencia", tabla["prevalencia"]),
    ]

    titulo("MEDIDAS DE CONCENTRACIÓN ENTRE ENTIDADES")
    print(f"\nEntidades: {n}. Gini máximo posible, (n-1)/n: {(n - 1) / n:.4f}")
    print(f"\n    {'Variable':<22}{'Gini':>10}")
    print(f"    {'-' * 32}")
    for nombre, valores in series:
        print(f"    {nombre:<22}{gini(valores):>10.4f}")

    casos, poblacion = (curva_lorenz(valores)[1] for _, valores in series[:2])
    print(f"\n    {'Parte del total que reúnen':<34}{'Casos':>10}{'Población':>12}")
    print(f"    {'-' * 56}")
    print(
        f"    {f'Las {n // 2} entidades con menos':<34}{100 * casos[n // 2]:>9.1f}%"
        f"{100 * poblacion[n // 2]:>11.1f}%"
    )
    print(
        f"    {'Las 5 entidades con más':<34}{100 * (1 - casos[n - 5]):>9.1f}%"
        f"{100 * (1 - poblacion[n - 5]):>11.1f}%"
    )
    print("    Cada columna ordena las entidades según su propio valor.")


# --------------------------------------------------------------------------
# Comprobación
# --------------------------------------------------------------------------


def gini_por_area(valores) -> float:
    """Devuelve 1 menos el doble del área bajo la curva de Lorenz."""
    unidades, total = curva_lorenz(valores)
    return float(1 - np.sum(np.diff(unidades) * (total[1:] + total[:-1])))


def exigir(nombre: str, obtenido: float, esperado: float) -> None:
    """Imprime el caso y lanza AssertionError si se aleja del valor esperado."""
    if abs(obtenido - esperado) > TOLERANCIA:
        raise AssertionError(f"{nombre}: se esperaba {esperado} y se obtuvo {obtenido}")
    print(f"    {nombre:<52}{obtenido:>10.6f}")


def comprobar() -> None:
    """
    Comprueba los índices contra casos de valor conocido y contra SciPy.

    El contraste con scipy.stats.entropy usa la población de cada categoría en
    el dataset procesado.
    """
    if not RUTA_ENDIREH_LIMPIO.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_ENDIREH_LIMPIO}. "
            "Ejecuta antes: python3 src/cleaning/limpieza.py"
        )

    una_categoria = [10, 0, 0, 0]
    uniforme = [5, 5, 5, 5]
    casos = [
        ("Una categoría con todo, Gini-Simpson", gini_simpson(una_categoria), 0),
        ("Una categoría con todo, IQV", iqv(una_categoria), 0),
        ("Una categoría con todo, entropía", entropia_shannon(una_categoria), 0),
        ("Cuatro categorías iguales, Gini-Simpson", gini_simpson(uniforme), 3 / 4),
        ("Cuatro categorías iguales, IQV", iqv(uniforme), 1),
        ("Cuatro categorías iguales, entropía", entropia_shannon(uniforme), 2),
        (
            "Cuatro categorías iguales, entropía normalizada",
            entropia_normalizada(uniforme),
            1,
        ),
        ("Seis valores iguales, Gini", gini([7] * 6), 0),
        ("Valores 1, 2, 3 y 4, Gini", gini([1, 2, 3, 4]), 1 / 4),
        (
            "Valores 1, 2, 3 y 4, Gini por área de Lorenz",
            gini_por_area([1, 2, 3, 4]),
            1 / 4,
        ),
        ("Todo en una de 6 unidades, Gini", gini([0] * 5 + [1]), 5 / 6),
        ("Todo en una de 32 unidades, Gini", gini([0] * 31 + [1]), 31 / 32),
    ]
    print("[indices] Casos de valor conocido")
    for nombre, obtenido, esperado in casos:
        exigir(nombre, obtenido, esperado)

    df = pl.read_parquet(RUTA_ENDIREH_LIMPIO)
    print("\n[indices] Entropía ponderada contra scipy.stats.entropy")
    for columna in VARIABLES_HETEROGENEIDAD:
        poblacion = tabla_frecuencias(df, columna)["poblacion"]
        de_scipy = float(entropy(poblacion.to_numpy(), base=2))
        exigir(columna, entropia_shannon(poblacion), de_scipy)

    total = len(casos) + len(VARIABLES_HETEROGENEIDAD)
    print(f"\n[indices] Listo: {total} comprobaciones correctas.")


if __name__ == "__main__":
    comprobar()
