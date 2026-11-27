"""Figuras del proyecto.

Responsabilidades (secciones 3.8 y 3.9 de los lineamientos):
- Figuras del análisis exploratorio (artifacts/btcusdt_eda.pdf).
- Curvas de entrenamiento por fold con anotaciones (artifacts/training_curves_fold_k.png).
- Curvas de capital en prueba contra ambos benchmarks (artifacts/equity_curves_test_fold_k.png).
- Acciones del agente y precio durante el periodo de mayor drawdown.
- Animación interactiva del agente operando en el periodo de prueba.

Las cifras se calculan en data.py y evaluate.py; aquí solo se grafican. Las figuras se crean con
matplotlib.figure.Figure y no con pyplot, así que no abren ventanas ni dependen del backend.
"""
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator, PercentFormatter

import data

# Paleta Okabe-Ito, verificada para daltonismo (protanopia, deuteranopia y tritanopia) y visión
# normal. Cada fold conserva su color, tipo de línea y marcador en todas las figuras, así que el
# color nunca es la única forma de distinguirlos.
COLORES_FOLD = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#D55E00"}
LINEAS_FOLD = {1: "-", 2: "--", 3: "-.", 4: ":"}
MARCADORES_FOLD = {1: "o", 2: "s", 3: "^", 4: "D"}
COLORES_TENDENCIA = {"alcista": "#0072B2", "bajista": "#D55E00"}
COLORES_PARTE = {"entrenamiento": "#9E9E9E", "validacion": "#E69F00", "prueba": "#0072B2"}
NOMBRES_PARTE = {"entrenamiento": "Entrenamiento", "validacion": "Validación", "prueba": "Prueba"}
NOMBRES_SERIE = {"log_precio": "log precio", "rendimiento": "r", "rendimiento_absoluto": "|r|"}
FONDO_CONCLUSION = {"estacionaria": "#CCE3F0", "persistente": "#F9E3B8", "raiz_unitaria": "#F5CFBB",
                    "indeterminada": "#EEEEEE"}
NOMBRES_CONCLUSION = {"estacionaria": "estacionaria", "persistente": "persistente", "raiz_unitaria": "raíz unitaria",
                      "indeterminada": "indeterminada"}
VOLATILIDADES = ["vol_24h", "vol_168h", "parkinson_24h", "sigma_ewma"]
GRIS = "#6B6B6B"
TAMANO_PAGINA = (11.69, 8.27)  # A4 horizontal, en pulgadas


def _p(valor: float) -> str:
    """Formato de un valor p."""
    return "< 0.001" if valor < 0.001 else f"{valor:.3f}"


def _rango(valores, formato: str, conectores: tuple[str, str] = ("de", "a")) -> str:
    """'de x a y' con el mínimo y el máximo, o solo 'x' si con el formato se ven iguales."""
    bajo, alto = formato.format(min(valores)), formato.format(max(valores))
    return bajo if bajo == alto else f"{conectores[0]} {bajo} {conectores[1]} {alto}"


def _lista(elementos: list[str]) -> str:
    """'a', 'a y b' o 'a, b y c'."""
    return elementos[0] if len(elementos) == 1 else ", ".join(elementos[:-1]) + " y " + elementos[-1]


def _nueva_pagina(titulo: str) -> Figure:
    """Página A4 horizontal con su título."""
    fig = Figure(figsize=TAMANO_PAGINA)
    fig.suptitle(titulo, x=0.015, y=0.98, ha="left", fontsize=13, fontweight="bold")
    return fig


def _guardar(pdf: PdfPages, fig: Figure, lectura: str) -> None:
    """Escribe la lectura de la página al pie, acomoda los ejes y agrega la página al PDF."""
    texto = textwrap.fill("Lectura: " + lectura, width=175)
    renglones = texto.count("\n") + 1
    fig.text(0.015, 0.015, texto, ha="left", va="bottom", fontsize=9, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.03 + 0.024 * renglones, 1, 0.95))
    pdf.savefig(fig)


def _tabla(ax, tabla: pd.DataFrame, titulo: str, anchos: list[float] | None = None, fondos: dict | None = None,
           escala: float = 1.3) -> None:
    """Dibuja una tabla en un eje sin marco.

    `anchos` son las fracciones del ancho del eje para cada columna; `fondos` colorea celdas con
    {(fila, columna): color}, contando desde la primera fila de datos.
    """
    ax.axis("off")
    celdas = ax.table(cellText=tabla.astype(str).to_numpy(), colLabels=list(tabla.columns), colWidths=anchos,
                      loc="upper center", cellLoc="center")
    celdas.auto_set_font_size(False)
    celdas.set_fontsize(8)
    celdas.scale(1, escala)
    for (fila, columna), celda in celdas.get_celld().items():
        celda.set_edgecolor("#D0D0D0")
        if fila == 0:
            celda.set_facecolor("#EEEEEE")
            celda.set_text_props(fontweight="bold")
        elif fondos and (fila - 1, columna) in fondos:
            celda.set_facecolor(fondos[(fila - 1, columna)])
    ax.set_title(titulo, fontsize=10, loc="left")


def _pagina_datos(pdf: PdfPages, datos: pd.DataFrame, reporte: dict, folds: pd.DataFrame, regimenes: pd.DataFrame) -> None:
    """Limpieza del CSV y partición walk-forward de ventana expansiva."""
    fig = _nueva_pagina("Datos: limpieza del CSV horario de BTCUSDT y partición walk-forward")
    rejilla = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1, 1.3])
    ax = fig.add_subplot(rejilla[0, :])
    for _, fila in folds.iterrows():
        inicio, fin = mdates.date2num(fila["inicio"]), mdates.date2num(fila["fin"])
        ax.barh(fila["fold"], fin - inicio, left=inicio, height=0.6, color=COLORES_PARTE[fila["parte"]],
                edgecolor="white", linewidth=2)
        etiqueta = f"{fila['inicio']:%Y-%m} a {fila['fin']:%Y}" if fila["parte"] == "entrenamiento" else f"{fila['inicio']:%Y}"
        ax.text((inicio + fin) / 2, fila["fold"], etiqueta, ha="center", va="center", fontsize=8,
                color="white" if fila["parte"] == "prueba" else "black")
    fila_huecos = max(data.FOLDS) + 1
    huecos = datos.index[datos["horas_desde_anterior"] > 24]
    ax.vlines(mdates.date2num(huecos), fila_huecos - 0.3, fila_huecos + 0.3, color="black", linewidth=1)
    ax.set_yticks(list(data.FOLDS) + [fila_huecos], labels=[f"Fold {f}" for f in data.FOLDS] + ["Huecos > 24 h"])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.set_xlabel("Fecha (UTC)")
    ax.set_title("Ventana expansiva: el entrenamiento crece un año por fold", fontsize=10, loc="left")
    ax.legend(handles=[Patch(color=c, label=NOMBRES_PARTE[p]) for p, c in COLORES_PARTE.items()]
              + [Line2D([], [], color="black", linewidth=1, label="Hueco de más de 24 h")],
              loc="lower right", bbox_to_anchor=(1, 1), ncols=4, fontsize=8, frameon=False)

    limpieza = pd.DataFrame({
        "Paso": ["Filas leídas del CSV", "Barras con minuto distinto de 0, alineadas a su hora",
                 "Horas duplicadas eliminadas (Unix en otra unidad)", "Barras finales", "Primera barra (UTC)",
                 "Última barra (UTC)", "Horas faltantes (no se rellenan)", "Huecos de más de una hora", "Hueco más largo",
                 "Barras con volumen cero"],
        "Resultado": [f"{reporte['filas_leidas']:,}", f"{reporte['barras_desfasadas_alineadas']:,}",
                      f"{reporte['horas_duplicadas_eliminadas']:,}", f"{reporte['barras_finales']:,}",
                      reporte["primera_barra"][:16], reporte["ultima_barra"][:16], f"{reporte['horas_faltantes']:,}",
                      f"{reporte['huecos_mayores_a_1h']:,}", f"{reporte['hueco_maximo_horas']:.0f} h",
                      f"{reporte['barras_con_volumen_cero']:,}"],
    })
    _tabla(fig.add_subplot(rejilla[1, 0]), limpieza, "Limpieza (data.limpiar)", anchos=[0.7, 0.3])

    def periodo(fila: pd.Series) -> str:
        """El año si la parte cubre un año calendario completo; si no, sus fechas."""
        anio = fila["inicio"].year
        if fila["inicio"] == pd.Timestamp(f"{anio}-01-01") and fila["fin"] == pd.Timestamp(f"{anio}-12-31 23:00"):
            return str(anio)
        return f"{fila['inicio']:%Y-%m-%d} a {fila['fin']:%Y-%m-%d}"

    por_parte = {parte: folds[folds["parte"] == parte].set_index("fold") for parte in NOMBRES_PARTE}
    filas = [[f] + [valor for parte in NOMBRES_PARTE
                    for valor in (periodo(por_parte[parte].loc[f]), f"{por_parte[parte].loc[f, 'barras']:,}")]
             for f in data.FOLDS]
    particion = pd.DataFrame(filas, columns=["Fold"] + [n for nombre in NOMBRES_PARTE.values() for n in (nombre, "Barras")])
    _tabla(fig.add_subplot(rejilla[1, 1]), particion, "Partición: periodo y número de barras de cada parte",
           anchos=[0.07, 0.25, 0.1, 0.12, 0.1, 0.26, 0.1])

    fin_fold_1 = pd.Timestamp(f"{data.FOLDS[min(data.FOLDS)][0]}-12-31 23:00")
    completos = regimenes[regimenes["fin"] <= fin_fold_1]
    faltantes = (datos["horas_desde_anterior"] - 1).groupby(datos.index.year).sum()
    anios = sorted(faltantes.nlargest(2).index)
    partes = [f"la {NOMBRES_PARTE[p].lower()} del fold {f}" for f, anios_fold in data.FOLDS.items()
              for p, a in zip(("validacion", "prueba"), anios_fold[1:]) if a in anios]
    _guardar(pdf, fig, (
        f"de {reporte['filas_leidas']:,} filas quedan {reporte['barras_finales']:,} barras horarias. Faltan "
        f"{reporte['horas_faltantes']:,} horas en {reporte['huecos_mayores_a_1h']:,} huecos (el mayor de "
        f"{reporte['hueco_maximo_horas']:.0f} h); el {faltantes[anios].sum() / faltantes.sum():.0%} de ellas está en "
        f"{_lista([str(a) for a in anios])}, que son {_lista(partes)}. No se rellenan, porque inventar precios crearía "
        f"rendimientos falsos: el rendimiento después de un hueco se divide entre la raíz de las horas transcurridas y las "
        f"ventanas de las variables se miden en horas de calendario. Las tres partes de cada fold no se traslapan y van en "
        f"orden. Incluso el entrenamiento más corto (fold 1) contiene los regímenes 1 a {len(completos)} de la página "
        f"siguiente, alcistas y bajistas."
    ))


def _pagina_regimenes(pdf: PdfPages, entrenamiento: pd.DataFrame, regimenes: pd.DataFrame) -> None:
    """Precio en escala logarítmica con los regímenes sombreados y su tabla (pregunta 1)."""
    fig = _nueva_pagina(f"Pregunta 1. Regímenes del entrenamiento más largo, el del fold 4 "
                        f"({entrenamiento.index[0]:%Y-%m} a {entrenamiento.index[-1]:%Y-%m})")
    ax, ax_tabla = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 2]})
    for numero, fila in enumerate(regimenes.itertuples(), start=1):
        ax.axvspan(fila.inicio, fila.fin, color=COLORES_TENDENCIA[fila.tendencia], alpha=0.15, linewidth=0)
        ax.text(fila.inicio + (fila.fin - fila.inicio) / 2, 0.03, str(numero), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.plot(entrenamiento.index, entrenamiento["close"], color="black", linewidth=0.8)
    for fold, (fin, _, _) in data.FOLDS.items():
        if fold < max(data.FOLDS):
            x = pd.Timestamp(f"{fin}-12-31 23:00")
            ax.axvline(x, color=GRIS, linestyle="--", linewidth=0.8)
            ax.text(x, 0.97, f"fin del entrenamiento\ndel fold {fold} ", transform=ax.get_xaxis_transform(),
                    ha="right", va="top", fontsize=7, color=GRIS)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([2_000, 5_000, 10_000, 20_000, 50_000, 100_000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(entrenamiento.index[0], entrenamiento.index[-1])
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Precio de cierre (USDT, escala log.)")
    ax.legend(handles=[Line2D([], [], color="black", linewidth=0.8, label="Cierre horario")]
              + [Patch(color=c, alpha=0.3, label=f"Régimen {t}") for t, c in COLORES_TENDENCIA.items()]
              + [Line2D([], [], color=GRIS, linestyle="--", linewidth=0.8, label="Fin del entrenamiento de los folds 1 a 3")],
              loc="upper left", fontsize=8)
    tabla = pd.DataFrame({
        "N.º": range(1, len(regimenes) + 1),
        "Inicio": regimenes["inicio"].dt.strftime("%Y-%m-%d"),
        "Fin": regimenes["fin"].dt.strftime("%Y-%m-%d"),
        "Días": regimenes["dias"].round().astype(int),
        "Tendencia": regimenes["tendencia"],
        "Retorno": regimenes["retorno_total"].map("{:+,.0%}".format),
        "Volatilidad anual": regimenes["volatilidad_anual"].map("{:.0%}".format),
        "Máximo drawdown": regimenes["maximo_drawdown"].map("{:.0%}".format),
    })
    fondos = {(i, 4): COLORES_TENDENCIA[t] + "40" for i, t in enumerate(regimenes["tendencia"])}
    _tabla(ax_tabla, tabla, f"Un régimen termina cuando el precio se aleja más de {data.UMBRAL_REGIMEN:.0%} de su último "
           "máximo (alcista) o mínimo (bajista)", fondos=fondos)

    alcistas = regimenes[regimenes["tendencia"] == "alcista"]
    bajistas = regimenes[regimenes["tendencia"] == "bajista"]
    volatilidad = regimenes["volatilidad_anual"]
    _guardar(pdf, fig, (
        f"con el umbral de {data.UMBRAL_REGIMEN:.0%} el entrenamiento se divide en {len(regimenes)} regímenes: "
        f"{len(alcistas)} alcistas, que suben {_rango(alcistas['retorno_total'], '{:,.0%}')}, y {len(bajistas)} bajistas, que "
        f"caen {_rango(-bajistas['retorno_total'], '{:.0%}')} y duran {_rango(bajistas['dias'], '{:.0f}')} días. La "
        f"volatilidad anual va de {volatilidad.max():.0%} (régimen {volatilidad.idxmax() + 1}) a {volatilidad.min():.0%} "
        f"(régimen {volatilidad.idxmin() + 1}): el régimen cambia la escala del riesgo, por eso el estado mide la volatilidad "
        f"y la normalización se ajusta por fold. El primer régimen empieza con los datos y el último termina con ellos, no "
        f"con giros confirmados. Los giros se fechan viendo toda la trayectoria, así que describen el pasado y nunca entran "
        f"al estado."
    ))


def _pagina_distribucion(pdf: PdfPages, entrenamiento: pd.DataFrame, distribucion: pd.DataFrame) -> None:
    """Histograma de los rendimientos horarios contra una normal y tabla por fold."""
    fig = _nueva_pagina("Distribución de los rendimientos horarios en el entrenamiento")
    ax, ax_tabla = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 2]})
    r = data.rendimiento_horario(entrenamiento).dropna()
    media, desviacion = r.mean(), r.std()
    bordes = np.linspace(media - 8 * desviacion, media + 8 * desviacion, 162)
    ancho = bordes[1] - bordes[0]
    fuera = np.mean(np.abs(r - media) > 8 * desviacion)
    ax.hist(r, bins=bordes, weights=np.full(len(r), 1 / (len(r) * ancho)), color=COLORES_FOLD[4], alpha=0.75,
            label=f"Rendimientos horarios del fold 4 ({len(r):,}; {fuera:.2%} quedan fuera de ±8 desviaciones)")
    x = np.linspace(bordes[0], bordes[-1], 600)
    ax.plot(x, np.exp(-0.5 * ((x - media) / desviacion) ** 2) / (desviacion * np.sqrt(2 * np.pi)), color="black",
            linewidth=1.2, label="Normal con la misma media y desviación")
    for signo in (-1, 1):
        ax.axvline(media + signo * 3 * desviacion, color=GRIS, linestyle=":", linewidth=1)
    ax.text(media + 3 * desviacion, 0.97, " ±3 desviaciones", transform=ax.get_xaxis_transform(), va="top", fontsize=8,
            color=GRIS)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5 / (len(r) * ancho))
    ax.set_xlim(bordes[0], bordes[-1])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Rendimiento logarítmico horario")
    ax.set_ylabel("Densidad (escala log.)")
    ax.legend(loc="upper left", fontsize=8)
    tabla = pd.DataFrame({
        "Fold": distribucion.index.astype(int),
        "Observaciones": distribucion["observaciones"].map("{:,.0f}".format),
        "Volatilidad anual": distribucion["volatilidad_anual"].map("{:.1%}".format),
        "Sesgo": distribucion["sesgo"].map("{:.2f}".format),
        "Curtosis en exceso": distribucion["curtosis_exceso"].map("{:.1f}".format),
        "Cuantil 1 %": distribucion["cuantil_1"].map("{:.2%}".format),
        "Cuantil 99 %": distribucion["cuantil_99"].map("{:.2%}".format),
        "|z| > 3": distribucion["fraccion_mas_de_3_desviaciones"].map("{:.2%}".format),
        "Jarque-Bera p": distribucion["jarque_bera_p"].map(_p),
    })
    _tabla(ax_tabla, tabla, "Forma de la distribución en el entrenamiento de cada fold (en una normal, |z| > 3 ocurre en "
           "el 0.27 % de las horas)")

    extremos = distribucion["fraccion_mas_de_3_desviaciones"]
    jarque_bera = ("rechaza la normalidad en los cuatro folds" if (distribucion["jarque_bera_p"] < 0.05).all()
                   else "no rechaza la normalidad en todos los folds")
    _guardar(pdf, fig, (
        f"la curtosis en exceso es {_rango(distribucion['curtosis_exceso'], '{:.0f}')} (0 en una normal) y el sesgo es "
        f"{_rango(distribucion['sesgo'], '{:.2f}')}: las caídas extremas pesan más que las subidas. Las horas con movimientos "
        f"de más de 3 desviaciones son el {_rango(extremos, '{:.1%}')}, unas {extremos.mean() / 0.0027:.0f} veces más que en "
        f"una normal. Jarque-Bera {jarque_bera}. Con colas así un solo dato extremo dominaría la entrada de la red: por eso "
        f"las variables se recortan a ±{data.Normalizador.LIMITE:.0f} desviaciones al normalizar, y el riesgo se evalúa con "
        f"drawdown y Sortino además de la volatilidad."
    ))


def _pagina_autocorrelacion(pdf: PdfPages, autocorrelacion: pd.DataFrame, ljung_box: pd.DataFrame,
                            observaciones: pd.Series) -> None:
    """Autocorrelación de r y de |r| por fold, con la banda de 95 % de ruido blanco (pregunta 2)."""
    fig = _nueva_pagina("Pregunta 2. Autocorrelación de r y de |r| en el entrenamiento de cada fold")
    ejes = fig.subplots(1, 2)
    banda = 1.96 / np.sqrt(observaciones.min())
    for ax, serie, titulo, lugar in (
        (ejes[0], "rendimiento", "Rendimiento horario r: casi sin memoria", "lower right"),
        (ejes[1], "rendimiento_absoluto", "|r|: la volatilidad se agrupa y dura semanas", "upper right"),
    ):
        ax.axhspan(-banda, banda, color="#BDBDBD", alpha=0.6, linewidth=0,
                   label=f"Banda de 95 % sin autocorrelación (±{banda:.3f}, fold 1)")
        ax.axhline(0, color="black", linewidth=0.6)
        for fold in data.FOLDS:
            ax.plot(autocorrelacion.index, autocorrelacion[f"{serie}_fold_{fold}"], color=COLORES_FOLD[fold],
                    linestyle=LINEAS_FOLD[fold], linewidth=1.2, label=f"Fold {fold}")
        prueba = ljung_box[(ljung_box["serie"] == serie) & (ljung_box["rezagos"] == 24)]
        ax.set_title(f"{titulo}\nLjung-Box con 24 rezagos: Q {_rango(prueba['estadistico'], '{:,.0f}')}, "
                     f"p {_p(prueba['p'].max())} en los cuatro folds", fontsize=10)
        ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
        ax.set_xlim(0, autocorrelacion.index[-1] + 1)
        ax.set_xlabel("Rezago (horas)")
        ax.set_ylabel("Autocorrelación")
        ax.legend(loc=lugar, fontsize=8)

    r1 = [autocorrelacion[f"rendimiento_fold_{f}"].iloc[0] for f in data.FOLDS]
    absoluto = [autocorrelacion[f"rendimiento_absoluto_fold_{f}"] for f in data.FOLDS]
    ultimo = autocorrelacion[f"rendimiento_fold_{max(data.FOLDS)}"]
    principales = ultimo.abs().nlargest(2).index
    _guardar(pdf, fig, (
        f"r casi no tiene memoria: en el fold 4 los rezagos con más dependencia son el {principales[0]} "
        f"({ultimo[principales[0]]:.3f}) y el {principales[1]} ({ultimo[principales[1]]:.3f}), una ligera reversión de la "
        f"hora anterior y de la misma hora del día anterior; el rezago 1 va {_rango(r1, '{:.3f}')} según el fold. Con tantas "
        f"observaciones Ljung-Box rechaza el ruido blanco aun con dependencias así de pequeñas. |r| sí tiene memoria: "
        f"{_rango([a.iloc[0] for a in absoluto], '{:.2f}')} en el rezago 1 y todavía "
        f"{_rango([a.iloc[-1] for a in absoluto], '{:.2f}')} a 168 horas. La dirección del siguiente movimiento casi no se "
        f"anticipa, pero su tamaño sí: por eso el estado usa rendimientos de varios plazos y volatilidades de 24 y 168 horas, "
        f"Parkinson y EWMA, en lugar de una ventana de precios crudos."
    ))


def _pagina_estacionariedad(pdf: PdfPages, estacionariedad: pd.DataFrame, analisis: dict[int, pd.DataFrame]) -> None:
    """Pruebas ADF y KPSS del precio, los rendimientos y las variables del estado."""
    fig = _nueva_pagina("Estacionariedad: ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) al 5 %, en el entrenamiento")
    ax_series, ax_variables = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    tabla = pd.DataFrame({
        "Fold": estacionariedad["fold"],
        "Serie": estacionariedad["serie"].map(NOMBRES_SERIE),
        "ADF": estacionariedad["adf_estadistico"].map("{:.1f}".format),
        "ADF p": estacionariedad["adf_p"].map(_p),
        "KPSS": estacionariedad["kpss_estadistico"].map("{:.2f}".format),
        "KPSS p": estacionariedad["kpss_p"].map("{:.2f}".format),
        "Conclusión": estacionariedad["conclusion"].map(NOMBRES_CONCLUSION),
    })
    fondos = {(i, 6): FONDO_CONCLUSION[c] for i, c in enumerate(estacionariedad["conclusion"])}
    _tabla(ax_series, tabla, "Log precio, r y |r|", anchos=[0.09, 0.15, 0.12, 0.12, 0.12, 0.12, 0.2], fondos=fondos, escala=2.4)
    codigos = pd.DataFrame({f"Fold {f}": t.set_index("variable")["conclusion"] for f, t in analisis.items()})
    conclusiones = codigos.apply(lambda columna: columna.map(NOMBRES_CONCLUSION))
    conclusiones.insert(0, "Familia", analisis[min(analisis)].set_index("variable")["familia"])
    conclusiones = conclusiones.rename_axis("Variable").reset_index()
    fondos = {(i, j + 2): FONDO_CONCLUSION[codigos.iat[i, j]] for i in range(codigos.shape[0]) for j in range(codigos.shape[1])}
    _tabla(ax_variables, conclusiones, "Variables del estado: conclusión de las mismas pruebas por fold",
           anchos=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15], fondos=fondos, escala=2.15)
    ax_variables.legend(handles=[Patch(color=c, label=NOMBRES_CONCLUSION[n]) for n, c in FONDO_CONCLUSION.items()],
                        loc="lower center", ncols=4, fontsize=8)

    def resumen(serie: str) -> str:
        filas = estacionariedad[estacionariedad["serie"] == serie]
        if filas["conclusion"].nunique() == 1:
            return f"{NOMBRES_CONCLUSION[filas['conclusion'].iloc[0]]} en los cuatro folds"
        return ", ".join(f"{NOMBRES_CONCLUSION[c]} en el fold {f}" for f, c in zip(filas["fold"], filas["conclusion"]))

    precio_con_raiz = (estacionariedad.loc[estacionariedad["serie"] == "log_precio", "conclusion"] == "raiz_unitaria").all()
    siempre = lambda c: list(codigos.index[(codigos == c).all(axis=1)])
    con_raiz = list(codigos.index[(codigos == "raiz_unitaria").any(axis=1)])
    _guardar(pdf, fig, (
        f"log precio: {resumen('log_precio')}" + (", así que el estado nunca usa precios en nivel" if precio_con_raiz else "")
        + f"; r: {resumen('rendimiento')}; |r|: {resumen('rendimiento_absoluto')}. Persistente significa que ADF rechaza la "
        f"raíz unitaria pero KPSS rechaza la estacionariedad: la serie regresa a su media muy lento o cambia de nivel entre "
        f"regímenes, como la volatilidad. De las {len(conclusiones)} variables, {len(siempre('estacionaria'))} son "
        f"estacionarias y {len(siempre('persistente'))} persistentes en los cuatro folds; "
        + (f"tienen raíz unitaria en algún fold: {_lista(con_raiz)}. " if con_raiz else "ninguna tiene raíz unitaria. ")
        + "Por eso cada variable es un rendimiento, un cociente o una distancia relativa, y la normalización se ajusta con el "
        "entrenamiento de cada fold. KPSS solo reporta p entre 0.01 y 0.10."
    ))


def _pagina_horizontes(pdf: PdfPages, horizontes: pd.DataFrame, costo_por_lado: float) -> None:
    """Movimiento típico del precio por plazo frente al costo de ida y vuelta."""
    fig = _nueva_pagina("Movimiento típico del precio por plazo frente al costo de una operación")
    ax = fig.subplots()
    costo = 2 * costo_por_lado
    for fold in data.FOLDS:
        ax.plot(horizontes.index, horizontes[fold], color=COLORES_FOLD[fold], linestyle=LINEAS_FOLD[fold],
                marker=MARCADORES_FOLD[fold], linewidth=1.5, markersize=7, label=f"Fold {fold}")
    ax.axhline(costo, color="black", linestyle="--", linewidth=1, label=f"Costo de ida y vuelta ({costo:.2%})")
    ax.axvline(data.HORIZONTE_HORAS, color=GRIS, linewidth=0.8)
    ax.text(data.HORIZONTE_HORAS, 0.97, f" horizonte de tenencia: {data.HORIZONTE_HORAS} h", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color=GRIS)
    ultimo = horizontes[max(data.FOLDS)]
    ax.text(0.01, 0.075, "Costo como porcentaje del movimiento mediano del fold 4:", transform=ax.transAxes, fontsize=8)
    for h, valor in ultimo.items():
        ax.text(h, 0.025, f"{costo / valor:.0%}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(costo / 2.2, horizontes.to_numpy().max() * 1.3)
    ax.set_xticks(horizontes.index, labels=[f"{h} h" for h in horizontes.index])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.0025, 0.005, 0.01, 0.02, 0.04, 0.06]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Plazo (horas, escala log.)")
    ax.set_ylabel("Mediana de |rendimiento| en el plazo (escala log.)")
    ax.legend(loc="upper left", fontsize=9)

    proporcion = costo / horizontes
    una_hora = _rango(proporcion.loc[1], "{:.0%}", ("entre", "y"))
    horizonte = _rango(proporcion.loc[data.HORIZONTE_HORAS], "{:.0%}", ("entre", "y"))
    _guardar(pdf, fig, (
        f"el costo de ida y vuelta ({costo:.2%}) equivale a {una_hora} del movimiento mediano de una hora según el fold, y a "
        f"{horizonte} del de {data.HORIZONTE_HORAS} horas. Operando cada hora casi todo el movimiento esperado se iría en "
        f"costos; a {data.HORIZONTE_HORAS} horas el costo es una fracción pequeña y el agente todavía puede cambiar de "
        f"posición varias veces por semana. Por eso el horizonte de tenencia es de {data.HORIZONTE_HORAS} h y el análisis "
        f"de variables usa el rendimiento de ese mismo plazo."
    ))


def _matriz_calor(fig: Figure, ax, matriz: pd.DataFrame, titulo: str, divergente: bool, formato: str,
                  limite: float | None = None) -> None:
    """Mapa de calor con el valor escrito en cada celda."""
    limite = limite or float(np.nanmax(np.abs(matriz.to_numpy())))
    escala = Normalize(-limite if divergente else 0, limite)
    imagen = ax.imshow(matriz.to_numpy(), cmap="RdBu_r" if divergente else "Blues", norm=escala, aspect="auto")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz.iat[i, j]
            oscuro = abs(valor) / limite > 0.6 if divergente else escala(valor) > 0.6
            ax.text(j, i, formato.format(valor), ha="center", va="center", fontsize=7, color="white" if oscuro else "black")
    ax.set_title(titulo, fontsize=9)
    fig.colorbar(imagen, ax=ax, fraction=0.05, pad=0.02)


def _separar_familias(ax, vertical: bool = False) -> None:
    """Líneas blancas entre las familias de variables."""
    posicion = -0.5
    for variables in list(data.FAMILIAS.values())[:-1]:
        posicion += len(variables)
        ax.axhline(posicion, color="white", linewidth=2.5)
        if vertical:
            ax.axvline(posicion, color="white", linewidth=2.5)


def _pagina_variables(pdf: PdfPages, analisis: dict[int, pd.DataFrame]) -> None:
    """Relación de cada variable con la dirección y con el tamaño del movimiento siguiente, por fold."""
    h = data.HORIZONTE_HORAS
    fig = _nueva_pagina(f"Pregunta 2. Cada variable contra el rendimiento de las siguientes {h} horas, "
                        "en el entrenamiento de cada fold")
    ejes = fig.subplots(1, 3, sharey=True)
    familia = analisis[min(analisis)].set_index("variable")["familia"]
    matriz = lambda columna: pd.DataFrame({f"Fold {f}": t.set_index("variable")[columna] for f, t in analisis.items()})
    # Las dos correlaciones de Spearman comparten escala para que su tamaño se compare a simple vista.
    comun = float(max(matriz(c).abs().to_numpy().max() for c in ("spearman_objetivo", "spearman_objetivo_abs")))
    paneles = (
        ("spearman_objetivo", f"Spearman con r de {h} h\n(¿anticipa la dirección?)", True, "{:+.3f}", comun),
        ("spearman_objetivo_abs", f"Spearman con |r| de {h} h\n(¿anticipa el tamaño?)", True, "{:+.2f}", comun),
        ("info_mutua_objetivo_abs", f"Información mutua con |r| de {h} h\n(tamaño, cualquier forma de relación)", False,
         "{:.3f}", None),
    )
    for ax, (columna, titulo, divergente, formato, limite) in zip(ejes, paneles):
        _matriz_calor(fig, ax, matriz(columna), titulo, divergente, formato, limite)
        ax.set_xticks(range(len(analisis)), labels=[f"Fold {f}" for f in analisis])
        _separar_familias(ax)
    ejes[0].set_yticks(range(len(familia)), labels=[f"{v} ({f})" for v, f in familia.items()], fontsize=8)

    direccion = matriz("spearman_objetivo").abs().stack()
    (variable_d, fold_d), valor_d = direccion.idxmax(), direccion.max()
    tamano = matriz("spearman_objetivo_abs").loc[VOLATILIDADES]
    mutua = matriz("info_mutua_objetivo_abs").mean(axis=1)
    mutua_signo = matriz("info_mutua_objetivo").mean(axis=1)
    signo = (f"La información mutua con r con signo también es mayor en {mutua_signo.idxmax()} ({mutua_signo.max():.3f}) "
             "porque capta la dispersión del objetivo, no su dirección." if mutua_signo.idxmax() in VOLATILIDADES else
             f"La mayor información mutua con r con signo es la de {mutua_signo.idxmax()} ({mutua_signo.max():.3f}).")
    _guardar(pdf, fig, (
        ("ninguna variable anticipa bien la dirección" if valor_d < 0.1 else "la dirección se anticipa poco")
        + f": la mayor |Spearman| con r de {h} h es {valor_d:.3f} ({variable_d}, {fold_d.lower()}). El tamaño "
        + ("sí se anticipa" if tamano.min().min() >= 0.2 else "se anticipa poco")
        + f": las volatilidades ({_lista(VOLATILIDADES)}) tienen Spearman {_rango(tamano.stack(), '{:.2f}')} con |r| y "
        f"{mutua.idxmax()} tiene la mayor información mutua ({mutua.max():.3f} en promedio). {signo} Los objetivos de horas "
        "vecinas se traslapan, así que estas cifras comparan variables entre sí y no son pruebas de significancia. Detalle "
        "en artifacts/feature_analysis_fold_k.csv."
    ))


def _pagina_redundancia(pdf: PdfPages, correlacion: pd.DataFrame) -> None:
    """Correlación de Spearman entre las variables del estado, ordenadas por familia."""
    fig = _nueva_pagina("Redundancia entre las variables del estado: correlación de Spearman en el entrenamiento del fold 4")
    ax = fig.subplots()
    _matriz_calor(fig, ax, correlacion, "Las líneas blancas separan las familias: rendimientos, volatilidad, tendencia, "
                  "volumen y creencia", True, "{:+.2f}", limite=1.0)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(correlacion)), labels=correlacion.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(correlacion)), labels=correlacion.index, fontsize=8)
    _separar_familias(ax, vertical=True)

    familia = {v: f for f, lista in data.FAMILIAS.items() for v in lista}
    pares = correlacion.where(np.triu(np.ones(correlacion.shape, dtype=bool), k=1)).stack()
    altos = pares[pares.abs() >= 0.9].sort_values(key=abs, ascending=False)
    dentro = lambda nombre: pares[[familia[a] == familia[b] == nombre for a, b in pares.index]].abs().max()
    _guardar(pdf, fig, (
        "hay pares casi redundantes (|ρ| ≥ 0.9): " + _lista([f"{x} y {y} ({rho:.2f})" for (x, y), rho in altos.items()])
        + f". Dentro de la familia de rendimientos la mayor correlación es {dentro('rendimientos'):.2f}"
        + (", así que cada plazo aporta algo distinto" if dentro("rendimientos") < 0.5 else "")
        + f"; dentro de la de volatilidad llega a {dentro('volatilidad'):.2f}. Los pares redundantes miden lo mismo con "
        "estimadores distintos (rango alto-bajo, EWMA, rendimiento medio entre su error estándar); la red puede combinarlos, "
        "y la ablación por familia mide si alguno sobra."
    ))


def pdf_eda(datos: pd.DataFrame, reporte: dict, resumen: dict[str, pd.DataFrame], analisis: dict[int, pd.DataFrame],
            costo_por_lado: float, ruta: Path) -> Path:
    """Escribe artifacts/btcusdt_eda.pdf con el análisis exploratorio del entrenamiento (sección 3.8).

    Recibe el reporte de data.cargar_datos(), las tablas de data.resumen_eda() y las de
    data.analisis_variables() por fold, y el costo por lado de evaluate.py.
    """
    entrenamiento = data.particionar(datos, max(data.FOLDS))["entrenamiento"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(ruta) as pdf:
        _pagina_datos(pdf, datos, reporte, resumen["folds"], resumen["regimenes"])
        _pagina_regimenes(pdf, entrenamiento, resumen["regimenes"])
        _pagina_distribucion(pdf, entrenamiento, resumen["distribucion"])
        _pagina_autocorrelacion(pdf, resumen["autocorrelacion"], resumen["ljung_box"], resumen["distribucion"]["observaciones"])
        _pagina_estacionariedad(pdf, resumen["estacionariedad"], analisis)
        _pagina_horizontes(pdf, resumen["horizontes"], costo_por_lado)
        _pagina_variables(pdf, analisis)
        _pagina_redundancia(pdf, resumen["correlacion_variables"])
        informacion = pdf.infodict()
        informacion["Title"] = "BTCUSDT horario: análisis exploratorio del entrenamiento"
        informacion["Author"] = "Equipo 7, Microestructura y Sistemas de Trading, ITESO"
    return ruta
