# Funciones de visualización para el EDA.

# Todas reciben un DataFrame para poder llamarlas desde cualquier notebook.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
COLOR = "steelblue"


# Añade un título al gráfico.
def _titulo(ax, texto):
    ax.set_title(texto, fontsize=12, fontweight="bold")


# Muestra el porcentaje de valores faltantes por columna.
def plot_valores_nulos(df: pd.DataFrame, figsize=(9, 4)):
    pct = (df.isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=figsize)
    if pct.empty:
        ax.text(0.5, 0.5, "No hay valores nulos", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=pct.values, y=pct.index, ax=ax, color=COLOR)
        for i, v in enumerate(pct.values):
            ax.text(v, i, f" {v:.1f}%", va="center")
        ax.set_xlabel("% de nulos")
        _titulo(ax, "Valores faltantes por columna")
    plt.tight_layout()
    return fig


# Muestra el porcentaje de ceros por columna.
# En este dataset un 0 suele ser un dato no informado.
def plot_ceros(df: pd.DataFrame, cols=("baths", "bedrooms", "price"), figsize=(8, 4)):
    cols = [c for c in cols if c in df.columns]
    pct = pd.Series({c: (df[c] == 0).mean() * 100 for c in cols}).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=pct.values, y=pct.index, ax=ax, color="darkorange")
    for i, v in enumerate(pct.values):
        ax.text(v, i, f" {v:.1f}%", va="center")
    ax.set_xlabel("% de filas con valor 0")
    _titulo(ax, "Ceros por columna (posibles datos no informados)")
    plt.tight_layout()
    return fig


# Muestra el histograma y boxplot de una variable numérica.
# log=True aplica la transformación log1p.
def plot_distribucion_numerica(df: pd.DataFrame, col: str, bins=40, log=False,
                               figsize=(12, 4)):
    serie = df[col].dropna()
    etiqueta = col
    if log:
        serie = np.log1p(serie[serie >= 0])
        etiqueta = f"log1p({col})"
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(serie, bins=bins, kde=True, ax=axes[0], color=COLOR)
    axes[0].set_xlabel(etiqueta)
    _titulo(axes[0], f"Distribución de {etiqueta}")
    sns.boxplot(x=serie, ax=axes[1], color=COLOR)
    axes[1].set_xlabel(etiqueta)
    _titulo(axes[1], f"Boxplot de {etiqueta}")
    plt.tight_layout()
    return fig


# Crea una cuadrícula de histogramas para varias variables numéricas.
def plot_distribuciones_numericas(df: pd.DataFrame, cols=None, ncols=3, bins=40):
    cols = cols or df.select_dtypes(include="number").columns.tolist()
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color=COLOR)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# Muestra las categorías más frecuentes y su porcentaje.
def plot_categorica(df: pd.DataFrame, col: str, top_n=15, figsize=(8, 5)):
    vc = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color=COLOR)
    total = len(df)
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v:,} ({v / total:.1%})", va="center")
    ax.set_xlabel("Nº de anuncios")
    _titulo(ax, f"Frecuencia de {col} (top {top_n})")
    plt.tight_layout()
    return fig


# Muestra una matriz triangular de correlaciones.
def plot_correlaciones(df: pd.DataFrame, cols=None, figsize=(9, 7), annot=True,
                       method="spearman"):
    num = df[cols] if cols else df.select_dtypes(include="number")
    corr = num.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax)
    _titulo(ax, f"Matriz de correlación ({method})")
    plt.tight_layout()
    return fig


# Muestra un boxplot de una variable numérica según una categoría.
def plot_boxplot_por_categoria(df: pd.DataFrame, cat: str, num: str, top_n=10,
                               log=True, figsize=(10, 5)):
    orden = df[cat].value_counts().head(top_n).index
    datos = df[df[cat].isin(orden)]
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=datos, x=cat, y=num, order=orden, ax=ax, color=COLOR)
    if log:
        ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    _titulo(ax, f"{num} por {cat}" + (" (escala log)" if log else ""))
    plt.tight_layout()
    return fig


# Muestra la relación entre dos variables mediante un gráfico de dispersión.
# Utiliza una muestra aleatoria para agilizar el gráfico.
def plot_scatter(df: pd.DataFrame, x: str, y: str, hue=None, muestra=20000,
                 logx=False, logy=False, figsize=(8, 5)):
    datos = df.sample(min(muestra, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=datos, x=x, y=y, hue=hue, alpha=0.3, s=12, ax=ax)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _titulo(ax, f"{y} vs {x}")
    plt.tight_layout()
    return fig


# Calcula y muestra el porcentaje de outliers por variable numérica.
# Utiliza la regla de 1.5 veces el rango intercuartílico (IQR).
def plot_outliers_iqr(df: pd.DataFrame, cols=None, figsize=(9, 4)):
    num = df[cols] if cols else df.select_dtypes(include="number")
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    pct = (((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).mean() * 100)
    pct = pct[pct > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    if pct.empty:
        ax.text(0.5, 0.5, "No se detectan outliers", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=pct.values, y=pct.index, ax=ax, color="indianred")
        for i, v in enumerate(pct.values):
            ax.text(v, i, f" {v:.1f}%", va="center")
        ax.set_xlabel("% de outliers (IQR)")
        _titulo(ax, "Outliers por variable")
    plt.tight_layout()
    return fig


# Muestra la ubicación geográfica de los anuncios mediante longitud y latitud.
# Permite detectar coordenadas erróneas.
def plot_mapa_coordenadas(df: pd.DataFrame, hue="city", muestra=30000, figsize=(7, 7)):
    datos = df.sample(min(muestra, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=datos, x="longitude", y="latitude", hue=hue, s=6,
                    alpha=0.5, ax=ax)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    _titulo(ax, "Ubicación de los anuncios")
    plt.tight_layout()
    return fig


# Muestra el número de anuncios publicados por periodo.
# Convierte la fecha al formato indicado si está almacenada como texto.
def plot_serie_temporal(df: pd.DataFrame, fecha="date_added", freq="M",
                        formato="%m-%d-%Y", figsize=(10, 4)):
    f = pd.to_datetime(df[fecha], format=formato, errors="coerce")
    s = f.dropna().dt.to_period(freq).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index.astype(str), s.values, marker="o", color=COLOR)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Nº de anuncios")
    _titulo(ax, "Anuncios publicados por periodo")
    plt.tight_layout()
    return fig
