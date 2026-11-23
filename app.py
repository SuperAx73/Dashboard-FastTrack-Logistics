import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Configuración general y estilos
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FastTrack Logistics | Eficiencia de Combustible",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    .header-box {
        border-left: 4px solid #1E3A8A;
        padding: 0.4rem 0 0.4rem 1rem;
        margin-bottom: 1.2rem;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #475569;
        margin: 0.3rem 0 0 0;
    }
    .decision-card {
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1E3A8A;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        background-color: #FFFFFF;
    }
    .decision-q {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }
    .decision-a {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.35rem;
    }
    .decision-why {
        font-size: 0.9rem;
        color: #334155;
        margin: 0;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETA_RUTAS = {
    "Autopista": "#1D4ED8",  # Azul corporativo
    "Urbana": "#64748B",     # Gris pizarra
    "Montaña": "#B91C1C",    # Rojo ladrillo
}
ORDEN_RUTAS = ["Autopista", "Urbana", "Montaña"]

VARS_MAP = {
    "Consumo de Combustible (L)": ("Consumo_Litros", "Litros"),
    "Peso de Carga (Ton)": ("Peso_Carga_Ton", "Toneladas"),
    "Velocidad Promedio (km/h)": ("Velocidad_Promedio_kmh", "km/h"),
}


@st.cache_data
def cargar_datos():
    for enc in ("cp850", "utf-8-sig", "latin1"):
        try:
            df_tmp = pd.read_csv("datos_logistica_transporte.csv", encoding=enc)
            if "Montaña" in df_tmp["Tipo_Ruta"].values:
                return df_tmp
        except UnicodeDecodeError:
            continue
    return pd.read_csv("datos_logistica_transporte.csv", encoding="cp850")


def obtener_moda(serie):
    conteos = serie.value_counts()
    if len(conteos) == 0 or conteos.max() <= 1:
        return "Sin moda"
    modas = conteos[conteos == conteos.max()].index.tolist()
    return ", ".join(f"{m:.1f}" for m in modas)


def tabla_descriptiva(data, col):
    filas = []
    for ruta in ORDEN_RUTAS:
        s = data[data["Tipo_Ruta"] == ruta][col]
        if len(s) == 0:
            continue
        filas.append({
            "Tipo de Ruta": ruta,
            "n": int(len(s)),
            "Media": round(s.mean(), 2),
            "Mediana": round(s.median(), 2),
            "Varianza": round(s.var(ddof=1), 2) if len(s) > 1 else 0.0,
            "Desv. Est.": round(s.std(ddof=1), 2) if len(s) > 1 else 0.0,
            "Mín.": round(s.min(), 2),
            "Máx.": round(s.max(), 2),
            "Rango": round(s.max() - s.min(), 2),
            "Moda": obtener_moda(s),
        })
    return pd.DataFrame(filas)


def calcular_anova(data, col):
    rutas_activas = [r for r in ORDEN_RUTAS if len(data[data["Tipo_Ruta"] == r]) > 0]
    grupos = [data[data["Tipo_Ruta"] == r][col].values for r in rutas_activas]
    n_total = len(data)
    k = len(grupos)
    media_g = data[col].mean()

    sc_entre = sum(len(g) * (g.mean() - media_g) ** 2 for g in grupos)
    sc_dentro = sum(((g - g.mean()) ** 2).sum() for g in grupos)
    sc_total = sc_entre + sc_dentro
    gl_entre = k - 1
    gl_dentro = n_total - k
    cm_entre = sc_entre / gl_entre
    cm_dentro = sc_dentro / gl_dentro
    f_stat, p_val = stats.f_oneway(*grupos)
    eta2 = sc_entre / sc_total if sc_total > 0 else 0.0
    _, p_levene = stats.levene(*grupos)

    tabla = pd.DataFrame([
        {
            "Fuente": "Entre rutas",
            "SC": f"{sc_entre:.4f}",
            "gl": gl_entre,
            "CM": f"{cm_entre:.4f}",
            "F": f"{f_stat:.4f}",
            "Valor-p": f"{p_val:.3e}",
        },
        {
            "Fuente": "Dentro de rutas",
            "SC": f"{sc_dentro:.4f}",
            "gl": gl_dentro,
            "CM": f"{cm_dentro:.4f}",
            "F": "-",
            "Valor-p": "-",
        },
        {
            "Fuente": "Total",
            "SC": f"{sc_total:.4f}",
            "gl": n_total - 1,
            "CM": "-",
            "F": "-",
            "Valor-p": "-",
        },
    ])
    return {
        "tabla": tabla,
        "F": f_stat,
        "p": p_val,
        "eta2": eta2,
        "p_levene": p_levene,
    }


def barra_filtros(df_origen, clave_prefijo):
    with st.container(border=True):
        f1, f2 = st.columns([1.2, 1.0], gap="large")
        with f1:
            rutas_sel = st.multiselect(
                "Filtrar por tipo de ruta",
                options=ORDEN_RUTAS,
                default=ORDEN_RUTAS,
                key=f"{clave_prefijo}_rutas",
            )
        with f2:
            p_min = float(np.floor(df_origen["Peso_Carga_Ton"].min()))
            p_max = float(np.ceil(df_origen["Peso_Carga_Ton"].max()))
            rango_p = st.slider(
                "Filtrar por peso de carga (Ton)",
                min_value=p_min,
                max_value=p_max,
                value=(p_min, p_max),
                step=0.5,
                key=f"{clave_prefijo}_peso",
            )
    df_out = df_origen[
        (df_origen["Tipo_Ruta"].isin(rutas_sel))
        & (df_origen["Peso_Carga_Ton"] >= rango_p[0])
        & (df_origen["Peso_Carga_Ton"] <= rango_p[1])
    ].copy()
    return df_out


df = cargar_datos()

# Modelos globales (N = 50)
X_base = sm.add_constant(df[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]])
y_all = df["Consumo_Litros"]
modelo_base = sm.OLS(y_all, X_base).fit()
y_pred_base = modelo_base.predict(X_base)

X_std = (df[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]] - df[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]].mean()) / df[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]].std()
y_std = (y_all - y_all.mean()) / y_all.std()
betas_std_base = sm.OLS(y_std, X_std).fit().params

modelo_ext = ols(
    "Consumo_Litros ~ Peso_Carga_Ton + Velocidad_Promedio_kmh + C(Tipo_Ruta, Treatment(reference='Autopista'))",
    data=df,
).fit()
y_pred_ext = modelo_ext.predict(df)

# ------------------------------------------------------------------
# Barra lateral: Navegación y ficha técnica
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### FASTTRACK LOGISTICS")
    st.caption("Estadística Aplicada a la Investigación · UACH")
    st.divider()

    seccion = st.radio(
        "Secciones del reporte",
        options=[
            "Resumen y Decisiones",
            "Descriptivas y ANOVA",
            "Análisis de Correlación",
            "Regresión Lineal Múltiple",
            "Simulador Operativo",
        ],
    )

    st.divider()
    st.markdown(
        "**Alumno:** Axiel Urenda  \n"
        "**Posgrado:** Maestría en Ciencias Computacionales  \n"
        "**Matrícula:** 360262  \n"
        f"**Registros:** {len(df)} viajes  \n"
        "**Archivo:** `datos_logistica_transporte.csv`"
    )

# ------------------------------------------------------------------
# Encabezado general
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <p class="header-title">Optimización de Rutas y Eficiencia de Combustible</p>
        <p class="header-sub">
            Análisis inferencial y modelado predictivo del consumo de diésel para contener el incremento del 18% en costos operativos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================================
# VISTA 1: RESUMEN Y DECISIONES
# ==================================================================
if seccion == "Resumen y Decisiones":
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Viajes analizados", f"{len(df)}")
    with k2:
        with st.container(border=True):
            st.metric("Consumo medio global", f"{df['Consumo_Litros'].mean():.2f} L", delta=f"DE: {df['Consumo_Litros'].std():.2f} L", delta_color="off")
    with k3:
        with st.container(border=True):
            st.metric("Consumo en Montaña", "75.01 L", delta="+33.76 L vs. Autopista", delta_color="inverse")
    with k4:
        with st.container(border=True):
            st.metric("Pronóstico (15 t, 80 km/h)", "44.76 L", delta="IC 95%: [41.48, 48.03] L", delta_color="off")

    col_chart, col_exec = st.columns([1.45, 1.0], gap="large")

    with col_chart:
        st.subheader("Impacto del Nivel de Carga por Tipo de Ruta")
        df_tiers = df.copy()
        df_tiers["Nivel_Carga"] = pd.cut(
            df_tiers["Peso_Carga_Ton"],
            bins=[0, 12, 20, 100],
            labels=["Ligera (<12 t)", "Media (12-20 t)", "Pesada (>20 t)"],
        )
        pivot_tiers = df_tiers.pivot_table(
            index="Nivel_Carga",
            columns="Tipo_Ruta",
            values="Consumo_Litros",
            aggfunc="mean",
            observed=False,
        )[ORDEN_RUTAS]

        fig_t, ax_t = plt.subplots(figsize=(7.5, 4.2))
        y_pos = np.arange(len(pivot_tiers.index))
        alto = 0.25

        for idx, ruta in enumerate(ORDEN_RUTAS):
            vals_t = pivot_tiers[ruta].values
            offset = (idx - 1) * alto
            bars = ax_t.barh(
                y_pos + offset,
                vals_t,
                height=alto,
                label=ruta,
                color=PALETA_RUTAS[ruta],
                alpha=0.9,
            )
            for bar, v in zip(bars, vals_t):
                ax_t.text(
                    v + 1.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} L",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#0F172A",
                )

        ax_t.set_yticks(y_pos)
        ax_t.set_yticklabels(pivot_tiers.index, fontsize=9.5)
        ax_t.set_xlabel("Consumo Promedio (Litros)")
        ax_t.set_xlim(0, pivot_tiers.values.max() * 1.18)
        ax_t.invert_yaxis()
        ax_t.grid(axis="x", linestyle="--", alpha=0.35)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        ax_t.legend(title="Tipo de Ruta", frameon=True, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

    with col_exec:
        st.subheader("Síntesis Gerencial")
        with st.container(border=True):
            st.markdown(
                "**Causa raíz del sobrecosto:** La combinación de rutas de Montaña con cargas superiores "
                "a 20 toneladas dispara el consumo hasta 84.87 L promedio (máximo 90.40 L)."
            )
            st.markdown(
                "**Palanca principal de ahorro:** El peso de carga explica la mayor parte del consumo "
                "individual (r = 0.8428), seguido de la topografía de la ruta (Eta cuadrado = 62.68%)."
            )
            st.markdown(
                "**Plan de acción sugerido:**\n"
                "1. Desviar cargas mayores a 20 toneladas hacia Autopista.\n"
                "2. Consolidar y balancear el tonelaje en despachos de Montaña.\n"
                "3. Presupuestar diésel con el modelo ajustado por ruta (R² = 97.48%)."
            )

    st.subheader("Toma de Decisiones y Sustento Estadístico")

    decisiones = [
        (
            "1. ¿Debería la empresa evitar ciertas rutas para cargas pesadas?",
            "Sí. Evitar la ruta de Montaña para cargas pesadas (> 20 toneladas).",
            "El ANOVA de peso (F = 8.2187, p = 8.693e-4) revela que Montaña concentra actualmente las cargas más pesadas (20.32 Ton en promedio). En Montaña cada tonelada adicional añade 2.27 L/Ton (r = 0.9963) frente a 1.30 L/Ton en Autopista. Los viajes de 20 o más toneladas promedian 84.87 L en Montaña contra 54.45 L en Autopista: reasignarlos ahorra ~30.4 L por viaje (~36%).",
        ),
        (
            "2. ¿En qué tipo de ruta se gasta más combustible y es estadísticamente significativa la diferencia?",
            "En Montaña (75.01 L por viaje en promedio). La diferencia es altamente significativa.",
            "El ANOVA de un factor arroja F(2, 47) = 39.4650 con p = 8.735e-11 (< 0.05), explicando el 62.68% de la varianza del consumo y cumpliendo homogeneidad de varianzas (Levene p = 0.0976). La prueba Post-Hoc Tukey HSD confirma diferencias significativas de Montaña frente a Autopista (+33.76 L, p < 0.0001) y Urbana (+32.75 L, p < 0.0001), mientras que Autopista y Urbana no difieren entre sí (+1.01 L, p = 0.9654).",
        ),
        (
            "3. ¿Enfocarse más en controlar la velocidad o en reducir el peso de carga?",
            "Priorizar la reducción y optimización del peso de carga.",
            "El peso tiene la correlación más alta con el consumo (r = 0.8428, p = 1.640e-14), mientras que la velocidad aislada no es significativa (r = -0.0939, p = 0.5167). En la regresión múltiple cada tonelada suma +2.4600 L (t = 14.0706, p = 1.809e-18) y su coeficiente estandarizado (0.9246) es 2.84 veces mayor que el de la velocidad (-0.3258). Al controlar por tipo de ruta, la velocidad deja de ser significativa (p = 0.0828).",
        ),
        (
            "4. Pronóstico para un camión con 15 toneladas viajando a 80 km/h",
            "Consumo estimado de 44.76 litros con el modelo principal (IC 95%: 41.48 a 48.03 L).",
            "Sustituyendo en la ecuación ajustada (26.1794 + 2.4600*15 - 0.2290*80) se obtienen 44.76 L con un R² de 80.98% y RMSE de 8.21 L. Como referencia operativa con el modelo extendido por ruta (R² = 97.48%, RMSE = 2.99 L), el mismo viaje estima 37.41 L en Autopista, 53.90 L en Urbana y 69.23 L en Montaña.",
        ),
    ]

    for pregunta, respuesta, porque in decisiones:
        st.markdown(
            f"""
            <div class="decision-card">
                <div class="decision-q">{pregunta}</div>
                <div class="decision-a">Decisión: {respuesta}</div>
                <p class="decision-why"><strong>Por qué:</strong> {porque}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================================================================
# VISTA 2: DESCRIPTIVAS Y ANOVA
# ==================================================================
elif seccion == "Descriptivas y ANOVA":
    df_f = barra_filtros(df, "anova")
    if len(df_f) == 0:
        st.warning("No hay registros con los filtros seleccionados.")
        st.stop()

    var_label = st.radio(
        "Variable cuantitativa a analizar",
        options=list(VARS_MAP.keys()),
        horizontal=True,
    )
    var_col, _ = VARS_MAP[var_label]

    grupos_validos = [r for r in ORDEN_RUTAS if len(df_f[df_f["Tipo_Ruta"] == r]) >= 2]
    if len(grupos_validos) >= 2:
        res_anova = calcular_anova(df_f[df_f["Tipo_Ruta"].isin(grupos_validos)], var_col)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.metric("Estadístico F", f"{res_anova['F']:.4f}")
        with m2:
            with st.container(border=True):
                st.metric("Valor-p", f"{res_anova['p']:.3e}")
        with m3:
            with st.container(border=True):
                st.metric("Varianza explicada (Eta²)", f"{res_anova['eta2'] * 100:.2f}%")
        with m4:
            with st.container(border=True):
                st.metric("Homogeneidad Levene (p)", f"{res_anova['p_levene']:.4f}")

        st.latex(r"H_0: \mu_{\text{Autopista}} = \mu_{\text{Urbana}} = \mu_{\text{Montaña}} \quad (\alpha = 0.05)")
        if res_anova["p"] < 0.05:
            st.info(
                f"Resultado: se rechaza la hipótesis nula (p = {res_anova['p']:.3e} < 0.05). "
                f"El tipo de ruta genera diferencias estadísticamente significativas en {var_label}."
            )
        else:
            st.info(f"Resultado: no se rechaza la hipótesis nula (p = {res_anova['p']:.4f}).")
    else:
        res_anova = None
        st.warning("Selecciona al menos dos rutas con dos o más viajes para calcular el ANOVA.")

    # Gráfico horizontal Violín + Caja + Puntos
    rutas_disp = [r for r in ORDEN_RUTAS if r in df_f["Tipo_Ruta"].unique()]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for idx, ruta in enumerate(rutas_disp):
        vals = df_f[df_f["Tipo_Ruta"] == ruta][var_col].values
        color = PALETA_RUTAS[ruta]

        if len(vals) >= 3:
            parts = ax.violinplot(vals, positions=[idx], vert=False, widths=0.62, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.18)

        ax.boxplot(
            vals,
            positions=[idx],
            vert=False,
            widths=0.22,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.45, edgecolor=color, linewidth=1.3),
            medianprops=dict(color="#0F172A", linewidth=1.8),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            showfliers=False,
        )
        rng = np.random.default_rng(100 + idx)
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(vals, np.full(len(vals), idx) + jitter, color=color, alpha=0.8, s=24, zorder=3)

        media_r = vals.mean()
        ax.scatter([media_r], [idx], marker="D", color="#0F172A", s=46, zorder=4, label="Media" if idx == 0 else None)
        ax.text(media_r, idx - 0.34, f"Media: {media_r:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")

    ax.set_yticks(range(len(rutas_disp)))
    ax.set_yticklabels([f"{r} (n={len(df_f[df_f['Tipo_Ruta'] == r])})" for r in rutas_disp], fontsize=9.5)
    ax.set_xlabel(var_label)
    ax.set_title(f"Distribución de {var_label} por Tipo de Ruta", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c_izq, c_der = st.columns(2, gap="large")
    with c_izq:
        st.markdown("#### Estadísticas Descriptivas")
        st.dataframe(tabla_descriptiva(df_f, var_col), hide_index=True, width="stretch")

        if res_anova is not None:
            st.markdown("#### Tabla ANOVA")
            st.dataframe(res_anova["tabla"], hide_index=True, width="stretch")

    with c_der:
        if len(grupos_validos) >= 2:
            st.markdown("#### Prueba Post-Hoc Tukey HSD")
            tukey = pairwise_tukeyhsd(
                endog=df_f[df_f["Tipo_Ruta"].isin(grupos_validos)][var_col],
                groups=df_f[df_f["Tipo_Ruta"].isin(grupos_validos)]["Tipo_Ruta"],
                alpha=0.05,
            )
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            st.dataframe(tukey_df, hide_index=True, width="stretch")

        st.markdown("#### Comparativo Global ANOVA (Dataset completo)")
        filas_res = []
        for lbl, (col_v, _) in VARS_MAP.items():
            ra = calcular_anova(df, col_v)
            filas_res.append({
                "Variable": lbl,
                "F": round(ra["F"], 4),
                "Valor-p": f"{ra['p']:.3e}",
                "Eta²": f"{ra['eta2'] * 100:.2f}%",
                "Decisión": "Rechazar H0" if ra["p"] < 0.05 else "No rechazar",
            })
        st.dataframe(pd.DataFrame(filas_res), hide_index=True, width="stretch")

    with st.expander("Inspeccionar tabla de registros filtrados"):
        st.dataframe(df_f, hide_index=True, width="stretch")

# ==================================================================
# VISTA 3: ANÁLISIS DE CORRELACIÓN
# ==================================================================
elif seccion == "Análisis de Correlación":
    df_f = barra_filtros(df, "corr")
    if len(df_f) < 3:
        st.warning("Se requieren al menos 3 viajes para calcular correlaciones.")
        st.stop()

    r_pc, p_pc = stats.pearsonr(df_f["Peso_Carga_Ton"], df_f["Consumo_Litros"])
    r_vc, p_vc = stats.pearsonr(df_f["Velocidad_Promedio_kmh"], df_f["Consumo_Litros"])
    r_pv, p_pv = stats.pearsonr(df_f["Peso_Carga_Ton"], df_f["Velocidad_Promedio_kmh"])

    top_l, top_r = st.columns([1.05, 1.0], gap="large")

    with top_l:
        cols_corr = ["Peso_Carga_Ton", "Velocidad_Promedio_kmh", "Consumo_Litros"]
        etiquetas = ["Peso (Ton)", "Velocidad (km/h)", "Consumo (L)"]
        mat_r = df_f[cols_corr].corr().values
        anot = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                if i == j:
                    anot[i, j] = "1.000"
                else:
                    r_ij, p_ij = stats.pearsonr(df_f[cols_corr[i]], df_f[cols_corr[j]])
                    p_txt = "p < 0.001" if p_ij < 0.001 else f"p = {p_ij:.3f}"
                    anot[i, j] = f"{r_ij:.3f}\n({p_txt})"

        fig_h, ax_h = plt.subplots(figsize=(5.5, 4.2))
        sns.heatmap(
            mat_r,
            annot=anot,
            fmt="",
            cmap="vlag",
            vmin=-1,
            vmax=1,
            xticklabels=etiquetas,
            yticklabels=etiquetas,
            linewidths=0.8,
            linecolor="#FFFFFF",
            annot_kws={"size": 9.5, "weight": "bold"},
            ax=ax_h,
        )
        ax_h.set_title("Matriz de Correlación de Pearson", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    with top_r:
        st.subheader("Interpretación de Asociaciones Lineales")
        with st.container(border=True):
            st.metric("Peso de Carga vs. Consumo (r)", f"{r_pc:.4f}", delta=f"p = {p_pc:.2e} | Relación muy fuerte", delta_color="off")
        with st.container(border=True):
            st.metric("Velocidad Promedio vs. Consumo (r)", f"{r_vc:.4f}", delta=f"p = {p_vc:.4f} | No significativa aislada", delta_color="off")
        with st.container(border=True):
            st.metric("Peso de Carga vs. Velocidad (r)", f"{r_pv:.4f}", delta=f"p = {p_pv:.4f} | Relación débil", delta_color="off")

        mostrar_tend = st.checkbox("Mostrar líneas de tendencia por tipo de ruta", value=True)

    st.divider()
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        fig_s1, ax_s1 = plt.subplots(figsize=(6.0, 4.2))
        for ruta in [r for r in ORDEN_RUTAS if r in df_f["Tipo_Ruta"].unique()]:
            sub = df_f[df_f["Tipo_Ruta"] == ruta]
            ax_s1.scatter(sub["Peso_Carga_Ton"], sub["Consumo_Litros"], label=ruta, color=PALETA_RUTAS[ruta], s=36, alpha=0.88)
            if mostrar_tend and len(sub) >= 2:
                m, b = np.polyfit(sub["Peso_Carga_Ton"], sub["Consumo_Litros"], 1)
                xs = np.linspace(sub["Peso_Carga_Ton"].min(), sub["Peso_Carga_Ton"].max(), 50)
                ax_s1.plot(xs, m * xs + b, linestyle="--", color=PALETA_RUTAS[ruta], linewidth=1.6)
        ax_s1.set_title("Consumo vs. Peso de Carga", fontweight="bold")
        ax_s1.set_xlabel("Peso de Carga (Toneladas)")
        ax_s1.set_ylabel("Consumo (Litros)")
        ax_s1.grid(linestyle="--", alpha=0.35)
        ax_s1.spines["top"].set_visible(False)
        ax_s1.spines["right"].set_visible(False)
        ax_s1.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig_s1)
        plt.close(fig_s1)

    with sc2:
        fig_s2, ax_s2 = plt.subplots(figsize=(6.0, 4.2))
        for ruta in [r for r in ORDEN_RUTAS if r in df_f["Tipo_Ruta"].unique()]:
            sub = df_f[df_f["Tipo_Ruta"] == ruta]
            ax_s2.scatter(sub["Velocidad_Promedio_kmh"], sub["Consumo_Litros"], label=ruta, color=PALETA_RUTAS[ruta], s=36, alpha=0.88)
            if mostrar_tend and len(sub) >= 2:
                m, b = np.polyfit(sub["Velocidad_Promedio_kmh"], sub["Consumo_Litros"], 1)
                xs = np.linspace(sub["Velocidad_Promedio_kmh"].min(), sub["Velocidad_Promedio_kmh"].max(), 50)
                ax_s2.plot(xs, m * xs + b, linestyle="--", color=PALETA_RUTAS[ruta], linewidth=1.6)
        ax_s2.set_title("Consumo vs. Velocidad Promedio", fontweight="bold")
        ax_s2.set_xlabel("Velocidad Promedio (km/h)")
        ax_s2.set_ylabel("Consumo (Litros)")
        ax_s2.grid(linestyle="--", alpha=0.35)
        ax_s2.spines["top"].set_visible(False)
        ax_s2.spines["right"].set_visible(False)
        ax_s2.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig_s2)
        plt.close(fig_s2)

# ==================================================================
# VISTA 4: REGRESIÓN LINEAL MÚLTIPLE
# ==================================================================
elif seccion == "Regresión Lineal Múltiple":
    modo_reg = st.radio(
        "Especificación del modelo (Mínimos Cuadrados Ordinarios, N = 50)",
        options=[
            "Modelo Base (Peso + Velocidad)",
            "Modelo Extendido (Peso + Velocidad + Tipo de Ruta)",
        ],
        horizontal=True,
    )

    es_ext = modo_reg.startswith("Modelo Extendido")

    if not es_ext:
        m_act = modelo_base
        y_hat = y_pred_base
        ecuacion = r"\widehat{Y} = 26.1794 + 2.4600\,X_1 - 0.2290\,X_2"
        nota_eq = "X1: Peso de carga (Ton) | X2: Velocidad promedio (km/h)."
        nombres_p = ["Intercepto", "Peso_Carga_Ton", "Velocidad_Promedio_kmh"]
        b_std_col = ["-", f"{betas_std_base['Peso_Carga_Ton']:.4f}", f"{betas_std_base['Velocidad_Promedio_kmh']:.4f}"]
    else:
        m_act = modelo_ext
        y_hat = y_pred_ext
        ecuacion = r"\widehat{Y} = 1.0277 + 1.7792\,X_1 + 0.1212\,X_2 + 16.4832\,D_{\text{Urbana}} + 31.8149\,D_{\text{Montaña}}"
        nota_eq = "Referencia base: Autopista. Al controlar por tipo de ruta, la velocidad pierde significancia (p = 0.0828)."
        nombres_p = [
            "Intercepto (Autopista)",
            "Ruta: Montaña",
            "Ruta: Urbana",
            "Peso_Carga_Ton",
            "Velocidad_Promedio_kmh",
        ]
        b_std_col = ["-"] * len(nombres_p)

    residuos = y_all - y_hat
    mse = float(np.mean(residuos ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuos)))

    col_eq, col_tab = st.columns([0.95, 1.35], gap="large")

    with col_eq:
        st.markdown("#### Ecuación y Rendimiento")
        with st.container(border=True):
            st.latex(ecuacion)
            st.caption(nota_eq)

        g_a, g_b = st.columns(2)
        with g_a:
            with st.container(border=True):
                st.metric("R²", f"{m_act.rsquared * 100:.2f}%", delta=f"Ajustado: {m_act.rsquared_adj * 100:.2f}%", delta_color="off")
            with st.container(border=True):
                st.metric("MSE", f"{mse:.2f} L²")
        with g_b:
            with st.container(border=True):
                st.metric("RMSE", f"{rmse:.2f} L", delta=f"MAE: {mae:.2f} L", delta_color="off")
            with st.container(border=True):
                st.metric("F Global", f"{m_act.fvalue:.2f}", delta=f"p = {m_act.f_pvalue:.2e}", delta_color="off")

    with col_tab:
        st.markdown("#### Validación de Coeficientes (Pruebas t, alfa = 0.05)")
        coef_df = pd.DataFrame({
            "Parámetro": nombres_p,
            "Coeficiente": [f"{v:.4f}" for v in m_act.params.values],
            "Error Est.": [f"{v:.4f}" for v in m_act.bse.values],
            "t": [f"{v:.4f}" for v in m_act.tvalues.values],
            "Valor-p": [f"{v:.3e}" if v < 0.001 else f"{v:.4f}" for v in m_act.pvalues.values],
            "Beta Est.": b_std_col,
            "Significativo": ["Sí" if p < 0.05 else "No" for p in m_act.pvalues.values],
        })
        st.dataframe(coef_df, hide_index=True, width="stretch")

    st.divider()
    d1, d2 = st.columns(2, gap="large")

    with d1:
        fig_rvp, ax_rvp = plt.subplots(figsize=(6.0, 4.1))
        for ruta in ORDEN_RUTAS:
            mask = df["Tipo_Ruta"] == ruta
            ax_rvp.scatter(y_all[mask], y_hat[mask], label=ruta, color=PALETA_RUTAS[ruta], s=36, alpha=0.88)
        lim_min = min(y_all.min(), y_hat.min()) - 4
        lim_max = max(y_all.max(), y_hat.max()) + 4
        ax_rvp.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1.2, label="Referencia ideal")
        ax_rvp.set_title(f"Consumo Real vs. Predicho (R² = {m_act.rsquared:.4f})", fontweight="bold")
        ax_rvp.set_xlabel("Consumo Real (Litros)")
        ax_rvp.set_ylabel("Consumo Predicho (Litros)")
        ax_rvp.grid(linestyle="--", alpha=0.35)
        ax_rvp.spines["top"].set_visible(False)
        ax_rvp.spines["right"].set_visible(False)
        ax_rvp.legend(frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_rvp)
        plt.close(fig_rvp)

    with d2:
        fig_res, ax_res = plt.subplots(figsize=(6.0, 4.1))
        for ruta in ORDEN_RUTAS:
            mask = df["Tipo_Ruta"] == ruta
            ax_res.scatter(y_hat[mask], residuos[mask], label=ruta, color=PALETA_RUTAS[ruta], s=36, alpha=0.88)
        ax_res.axhline(0, color="#0F172A", linestyle="--", linewidth=1.2)
        ax_res.set_title("Diagnóstico de Residuos vs. Valores Ajustados", fontweight="bold")
        ax_res.set_xlabel("Consumo Predicho (Litros)")
        ax_res.set_ylabel("Residuo (Litros)")
        ax_res.grid(linestyle="--", alpha=0.35)
        ax_res.spines["top"].set_visible(False)
        ax_res.spines["right"].set_visible(False)
        ax_res.legend(frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)

# ==================================================================
# VISTA 5: SIMULADOR OPERATIVO Y FINANCIERO
# ==================================================================
elif seccion == "Simulador Operativo":
    with st.container(border=True):
        st.markdown("#### Panel de Configuración del Escenario")
        r1_c1, r1_c2, r1_c3 = st.columns(3, gap="medium")
        with r1_c1:
            peso_sim = st.slider("Peso de carga (Toneladas)", 4.0, 30.0, 15.0, 0.5)
        with r1_c2:
            vel_sim = st.slider("Velocidad promedio (km/h)", 20.0, 110.0, 80.0, 1.0)
        with r1_c3:
            ruta_sim = st.selectbox("Tipo de ruta evaluada", ORDEN_RUTAS, index=0)

        r2_c1, r2_c2 = st.columns(2, gap="medium")
        with r2_c1:
            precio_diesel = st.number_input("Precio del diésel (MXN / Litro)", min_value=10.0, max_value=60.0, value=25.50, step=0.50)
        with r2_c2:
            viajes_mes = st.slider("Frecuencia mensual (viajes)", 10, 500, 100, 10)

    inp_base = pd.DataFrame({"const": [1.0], "Peso_Carga_Ton": [peso_sim], "Velocidad_Promedio_kmh": [vel_sim]})
    pred_sf = modelo_base.get_prediction(inp_base).summary_frame(alpha=0.05)
    cons_base = float(pred_sf["mean"].iloc[0])
    ic_low = float(pred_sf["mean_ci_lower"].iloc[0])
    ic_high = float(pred_sf["mean_ci_upper"].iloc[0])

    cons_rutas = {
        r: float(modelo_ext.predict(pd.DataFrame({
            "Peso_Carga_Ton": [peso_sim],
            "Velocidad_Promedio_kmh": [vel_sim],
            "Tipo_Ruta": [r],
        })).iloc[0])
        for r in ORDEN_RUTAS
    }
    cons_sel = cons_rutas[ruta_sim]
    costo_viaje = cons_sel * precio_diesel
    ahorro_litros = cons_rutas["Montaña"] - cons_sel
    ahorro_mensual = ahorro_litros * precio_diesel * viajes_mes

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        with st.container(border=True):
            st.metric("Modelo Base", f"{cons_base:.2f} L", delta=f"IC 95%: [{ic_low:.1f}, {ic_high:.1f}] L", delta_color="off")
    with s2:
        with st.container(border=True):
            st.metric(f"Ajustado ({ruta_sim})", f"{cons_sel:.2f} L", delta="R² = 97.48%", delta_color="off")
    with s3:
        with st.container(border=True):
            st.metric("Costo por viaje", f"${costo_viaje:,.0f} MXN", delta=f"${costo_viaje * viajes_mes:,.0f} MXN/mes", delta_color="off")
    with s4:
        with st.container(border=True):
            st.metric(
                f"Ahorro vs. Montaña ({viajes_mes} viajes)",
                f"${ahorro_mensual:,.0f} MXN",
                delta=f"{ahorro_litros:.1f} L menos por viaje" if ahorro_litros > 0 else "Ruta de mayor consumo",
                delta_color="normal" if ahorro_litros > 0 else "off",
            )

    p_izq, p_der = st.columns(2, gap="large")

    with p_izq:
        fig_bar, ax_bar = plt.subplots(figsize=(6.0, 4.0))
        vals_bar = [cons_rutas[r] for r in ORDEN_RUTAS]
        colores_bar = [PALETA_RUTAS[r] for r in ORDEN_RUTAS]
        bars = ax_bar.barh(ORDEN_RUTAS, vals_bar, color=colores_bar, alpha=0.9, height=0.52)
        for bar, v in zip(bars, vals_bar):
            ax_bar.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f} L (${v * precio_diesel:,.0f})", va="center", fontsize=9, fontweight="bold", color="#0F172A")
        ax_bar.set_xlim(0, max(vals_bar) * 1.32)
        ax_bar.invert_yaxis()
        ax_bar.set_title(f"Comparativa por Ruta ({peso_sim:.1f} t, {vel_sim:.0f} km/h)", fontweight="bold")
        ax_bar.set_xlabel("Consumo Estimado (Litros)")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.35)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with p_der:
        pesos_grid = np.linspace(4.0, 30.0, 60)
        y_grid_base = (
            modelo_base.params["const"]
            + modelo_base.params["Peso_Carga_Ton"] * pesos_grid
            + modelo_base.params["Velocidad_Promedio_kmh"] * vel_sim
        )
        y_grid_ruta = modelo_ext.predict(pd.DataFrame({
            "Peso_Carga_Ton": pesos_grid,
            "Velocidad_Promedio_kmh": np.full_like(pesos_grid, vel_sim),
            "Tipo_Ruta": [ruta_sim] * len(pesos_grid),
        }))

        fig_sens, ax_sens = plt.subplots(figsize=(6.0, 4.0))
        ax_sens.plot(pesos_grid, y_grid_base, linestyle="--", color="#334155", linewidth=1.6, label="Modelo Base")
        ax_sens.plot(pesos_grid, y_grid_ruta, color=PALETA_RUTAS[ruta_sim], linewidth=2.2, label=f"Modelo ({ruta_sim})")
        ax_sens.scatter([peso_sim], [cons_base], color="#334155", s=50, zorder=5)
        ax_sens.scatter([peso_sim], [cons_sel], color=PALETA_RUTAS[ruta_sim], s=58, zorder=5)
        ax_sens.set_title(f"Sensibilidad al Peso de Carga (a {vel_sim:.0f} km/h)", fontweight="bold")
        ax_sens.set_xlabel("Peso de Carga (Toneladas)")
        ax_sens.set_ylabel("Consumo Estimado (Litros)")
        ax_sens.grid(linestyle="--", alpha=0.35)
        ax_sens.spines["top"].set_visible(False)
        ax_sens.spines["right"].set_visible(False)
        ax_sens.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig_sens)
        plt.close(fig_sens)
