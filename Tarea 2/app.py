import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Predicción Inmobiliaria - Análisis y ML",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_NOMBRE = "usa-real-estate-dataset.zip"
CSV_NOMBRE = "realtor-data.zip.csv"
ZIP_RUTA = os.path.join(BASE_DIR, ZIP_NOMBRE)
CSV_RUTA = os.path.join(BASE_DIR, CSV_NOMBRE)

COLS_USO = [
    "price", "bed", "bath", "acre_lot",
    "city", "state", "zip_code", "house_size", "prev_sold_date"
]
FEATURES = [
    "superficie_m2",
    "terreno_m2",
    "bed",
    "bath",
    "tiene_historial_venta",
    "precio_mediano_estado",
    "precio_mediano_zip",
    "oferta_ciudad"
]


@st.cache_data(show_spinner="Cargando y limpiando dataset...")
def cargar_datos(ruta, es_zip, n_muestra):
    if es_zip:
        with zipfile.ZipFile(ruta, "r") as z:
            archivo_csv = [
                f for f in z.namelist()
                if f.endswith(".csv") and not f.startswith("__MACOSX")
            ][0]
            with z.open(archivo_csv) as f:
                df_raw = pd.read_csv(f, usecols=COLS_USO)
    else:
        archivo_csv = os.path.basename(ruta)
        df_raw = pd.read_csv(ruta, usecols=COLS_USO)

    total_raw = len(df_raw)
    nulos_raw = pd.DataFrame({
        "Variable": df_raw.columns,
        "Tipo Original": df_raw.dtypes.astype(str).values,
        "Valores Nulos": df_raw.isna().sum().values,
        "% Nulos": (df_raw.isna().mean().values * 100)
    }).sort_values(by="% Nulos", ascending=False).reset_index(drop=True)

    df = df_raw.dropna(
        subset=["price", "bed", "bath", "acre_lot", "house_size", "state", "city", "zip_code"]
    ).copy()
    despues_nulos = len(df)

    df = df[
        (df["price"].between(50_000, 2_000_000)) &
        (df["house_size"].between(300, 10_000)) &
        (df["bed"].between(1, 10)) &
        (df["bath"].between(1, 10)) &
        (df["acre_lot"].between(0.01, 20))
    ].copy()
    total_limpios = len(df)

    df["superficie_m2"] = df["house_size"] * 0.092903
    df["terreno_m2"] = df["acre_lot"] * 4046.86
    df["precio_m2"] = df["price"] / df["superficie_m2"]
    df["zip_code"] = df["zip_code"].astype(int).astype(str).str.zfill(5)
    df["anio_venta_previa"] = pd.to_datetime(df["prev_sold_date"], errors="coerce").dt.year
    df["tiene_historial_venta"] = df["anio_venta_previa"].notna().astype(int)

    if len(df) > n_muestra:
        df = df.sample(n=n_muestra, random_state=42).reset_index(drop=True)

    meta = {
        "total_raw": total_raw,
        "despues_nulos": despues_nulos,
        "total_limpios": total_limpios,
        "nulos_raw": nulos_raw,
        "archivo_interno": archivo_csv
    }
    return df, meta


@st.cache_resource(show_spinner="Entrenando modelos estadísticos y de Machine Learning...")
def entrenar_modelos(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.copy()
    test_df = test_df.copy()

    mediana_global = float(train_df["price"].median())
    mediana_estado = train_df.groupby("state")["price"].median().to_dict()
    mediana_zip = train_df.groupby("zip_code")["price"].median().to_dict()
    oferta_ciudad = train_df.groupby("city")["price"].count().to_dict()

    for subset in (train_df, test_df):
        subset["precio_mediano_estado"] = subset["state"].map(mediana_estado).fillna(mediana_global)
        subset["precio_mediano_zip"] = subset["zip_code"].map(mediana_zip).fillna(subset["precio_mediano_estado"])
        subset["oferta_ciudad"] = subset["city"].map(oferta_ciudad).fillna(1)

    X_train, y_train = train_df[FEATURES], train_df["price"]
    X_test, y_test = test_df[FEATURES], test_df["price"]

    ols_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    ols_table = pd.DataFrame({
        "Coeficiente": ols_model.params,
        "Error Estándar": ols_model.bse,
        "t-stat": ols_model.tvalues,
        "p-value": ols_model.pvalues,
        "Significativa (p < 0.05)": np.where(ols_model.pvalues < 0.05, "Sí", "No")
    })
    ols_info = {
        "r2": float(ols_model.rsquared),
        "r2_adj": float(ols_model.rsquared_adj),
        "f_stat": float(ols_model.fvalue),
        "f_pvalue": float(ols_model.f_pvalue),
        "fitted": ols_model.fittedvalues.values,
        "resid": ols_model.resid.values
    }

    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.07, max_depth=10, random_state=42
        )
    }

    metricas = []
    predicciones = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        predicciones[nombre] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metricas.append({
            "Modelo": nombre,
            "MSE": mse,
            "RMSE (USD)": rmse,
            "MAE (USD)": mae,
            "R2": r2
        })

    df_metricas = pd.DataFrame(metricas).sort_values(by="R2", ascending=False).reset_index(drop=True)
    importancias = pd.Series(
        modelos["Random Forest"].feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    referencias = {
        "mediana_global": mediana_global,
        "mediana_estado": mediana_estado,
        "mediana_zip": mediana_zip,
        "oferta_ciudad": oferta_ciudad
    }

    return modelos, df_metricas, ols_table, ols_info, importancias, y_test, predicciones, referencias


# --- Contenedor superior de credenciales en la barra lateral ---
credenciales_box = st.sidebar.container()
st.sidebar.divider()
st.sidebar.header("Configuración de Análisis")

n_muestra = st.sidebar.slider("Tamaño de muestra para análisis y entrenamiento", 5_000, 60_000, 25_000, step=5_000)

if os.path.exists(ZIP_RUTA):
    ruta_datos = ZIP_RUTA
    es_zip = True
    nombre_archivo = ZIP_NOMBRE
elif os.path.exists(CSV_RUTA):
    ruta_datos = CSV_RUTA
    es_zip = False
    nombre_archivo = CSV_NOMBRE
else:
    st.error(f"No se encontró '{ZIP_NOMBRE}' ni '{CSV_NOMBRE}' en el directorio del proyecto.")
    st.stop()

df, meta = cargar_datos(ruta_datos, es_zip, n_muestra)

estados_disponibles = sorted(df["state"].unique().tolist())
estados_sel = st.sidebar.multiselect("Filtrar estados en EDA (vacío = todos)", estados_disponibles)
df_eda = df[df["state"].isin(estados_sel)] if estados_sel else df

with credenciales_box:
    st.markdown(
        f"**Alumno:** Axiel Urenda  \n"
        f"**Posgrado:** Maestría en Ciencias Computacionales  \n"
        f"**Matrícula:** 360262  \n"
        f"**Registros:** {len(df_eda):,} propiedades  \n"
        f"**Archivo:**  \n"
        f"`{nombre_archivo}`"
    )

# --- Encabezado Principal ---
st.title("Predicción del Valor de Propiedades Inmobiliarias")
st.caption("Análisis estadístico descriptivo, correlación, regresión lineal y comparación de modelos de Machine Learning sobre el mercado inmobiliario de EE. UU.")

# KPIs principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros en Muestra", f"{len(df_eda):,}", f"De {meta['total_limpios']:,} limpios")
col2.metric("Precio Mediano", f"${df_eda['price'].median():,.0f}")
col3.metric("Superficie Mediana", f"{df_eda['superficie_m2'].median():,.1f} m2")
col4.metric("Precio Mediano por m2", f"${df_eda['precio_m2'].median():,.0f}")

# Entrenamiento de modelos
modelos, df_metricas, ols_table, ols_info, importancias, y_test, predicciones, refs = entrenar_modelos(df)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Análisis del Dataset",
    "2. Exploración y Correlación",
    "3. Modelo Estadístico (OLS)",
    "4. Comparación de Modelos ML",
    "5. Simulador de Precio",
    "6. Conclusiones y Recomendaciones"
])

# ==============================================================================
# TAB 1: ANÁLISIS DETALLADO Y DESCRIPTIVO DEL DATASET
# ==============================================================================
with tab0:
    st.subheader("Descripción General y Alcance del Dataset")
    st.markdown(
        "El conjunto de datos **USA Real Estate Dataset** recopila listados residenciales del mercado inmobiliario "
        "estadounidense. Incluye atributos físicos de cada inmueble, su localización geográfica (estado, ciudad y código postal) "
        "y registros de ventas previas. Para reflejar el entorno económico local sin incorporar fuentes externas inconsistentes, "
        "se construyen indicadores agregados por zona geográfica (precio mediano por código postal, precio mediano por estado "
        "y volumen de oferta por ciudad) calculados estrictamente sobre el conjunto de entrenamiento para evitar fuga de datos."
    )

    st.subheader("Auditoría de Limpieza y Preprocesamiento")
    c_a, c_b, c_c, c_d = st.columns(4)
    c_a.metric("Registros Originales", f"{meta['total_raw']:,}")
    c_b.metric("Tras Eliminar Nulos", f"{meta['despues_nulos']:,}", f"-{meta['total_raw'] - meta['despues_nulos']:,}")
    c_c.metric("Tras Filtrar Atípicos", f"{meta['total_limpios']:,}", f"-{meta['despues_nulos'] - meta['total_limpios']:,}")
    c_d.metric("Muestra Analizada", f"{len(df):,}", f"{(len(df) / meta['total_limpios'] * 100):.1f}% del limpio")

    st.markdown(
        "**Criterios de depuración aplicados:** se descartaron registros con valores faltantes en las variables estructurales "
        "y se acotaron valores atípicos extremos (precios entre 50,000 y 2,000,000 USD; superficie construida entre 300 y 10,000 ft2; "
        "habitaciones y baños entre 1 y 10; terreno entre 0.01 y 20 acres) con el fin de estabilizar la varianza y evitar distorsiones "
        "en la estimación de mínimos cuadrados ordinarios."
    )

    col_aud1, col_aud2 = st.columns(2)
    with col_aud1:
        st.markdown("**Diagnóstico de Valores Faltantes en el Dataset Original**")
        st.dataframe(
            meta["nulos_raw"].style.format({
                "Valores Nulos": "{:,.0f}",
                "% Nulos": "{:.2f}%"
            }),
            use_container_width=True
        )

    with col_aud2:
        st.markdown("**Diccionario de Variables Transformadas**")
        diccionario = pd.DataFrame([
            {"Variable": "price", "Tipo": "Numérica continua", "Unidad": "USD", "Rol": "Variable objetivo (Y)"},
            {"Variable": "superficie_m2", "Tipo": "Numérica continua", "Unidad": "m2", "Rol": "Predictora física (house_size * 0.092903)"},
            {"Variable": "terreno_m2", "Tipo": "Numérica continua", "Unidad": "m2", "Rol": "Predictora física (acre_lot * 4046.86)"},
            {"Variable": "bed / bath", "Tipo": "Numérica discreta", "Unidad": "Conteo", "Rol": "Predictoras físicas de distribución"},
            {"Variable": "tiene_historial_venta", "Tipo": "Binaria (0/1)", "Unidad": "Indicador", "Rol": "Dato histórico derivado de prev_sold_date"},
            {"Variable": "precio_mediano_zip", "Tipo": "Numérica continua", "Unidad": "USD", "Rol": "Indicador de entorno microeconómico"},
            {"Variable": "precio_mediano_estado", "Tipo": "Numérica continua", "Unidad": "USD", "Rol": "Indicador de entorno macroeconómico"},
            {"Variable": "oferta_ciudad", "Tipo": "Numérica discreta", "Unidad": "Conteo", "Rol": "Densidad de oferta del mercado local"}
        ])
        st.dataframe(diccionario, use_container_width=True, hide_index=True)

    st.subheader("Análisis Estadístico Descriptivo Detallado")
    cols_desc = ["price", "superficie_m2", "terreno_m2", "bed", "bath", "precio_m2", "tiene_historial_venta"]
    desc_base = df_eda[cols_desc].describe().T
    desc_base["IQR"] = desc_base["75%"] - desc_base["25%"]
    desc_base["CV (%)"] = (desc_base["std"] / desc_base["mean"]) * 100
    desc_base["Asimetría"] = df_eda[cols_desc].skew()
    desc_base["Curtosis"] = df_eda[cols_desc].kurt()
    desc_ordenado = desc_base[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "IQR", "CV (%)", "Asimetría", "Curtosis"]]
    desc_ordenado.columns = ["Conteo", "Media", "Desv. Est.", "Mínimo", "Q1 (25%)", "Mediana (50%)", "Q3 (75%)", "Máximo", "RIC", "CV (%)", "Asimetría", "Curtosis"]

    st.dataframe(desc_ordenado.style.format("{:,.2f}"), use_container_width=True)

    media_p = df_eda["price"].mean()
    mediana_p = df_eda["price"].median()
    asim_p = df_eda["price"].skew()
    cv_p = (df_eda["price"].std() / media_p) * 100
    pct_hist = df_eda["tiene_historial_venta"].mean() * 100

    st.info(
        f"Interpretación descriptiva: El precio promedio (${media_p:,.0f} USD) supera a la mediana (${mediana_p:,.0f} USD), "
        f"lo que confirma una asimetría positiva (coeficiente de asimetría = {asim_p:.2f}) típica del sector inmobiliario debido "
        f"a la concentración de propiedades de gama alta en la cola derecha. El coeficiente de variación del precio es de {cv_p:.1f}%, "
        f"reflejando alta heterogeneidad entre regiones. Adicionalmente, el {pct_hist:.1f}% de las propiedades analizadas cuenta con "
        f"un registro histórico de venta previa."
    )

    st.subheader("Cobertura Geográfica del Mercado")
    g1, g2, g3 = st.columns(3)
    g1.metric("Estados Representados", f"{df_eda['state'].nunique():,}")
    g2.metric("Ciudades Únicas", f"{df_eda['city'].nunique():,}")
    g3.metric("Códigos Postales (ZIP)", f"{df_eda['zip_code'].nunique():,}")

    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown("**Top 10 Estados con Mayor Precio Mediano (USD)**")
        top_precio = df_eda.groupby("state")["price"].median().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_precio.values, y=top_precio.index, ax=ax, color="#4c72b0")
        ax.set_xlabel("Precio Mediano (USD)")
        ax.set_ylabel("Estado")
        st.pyplot(fig)
        plt.close(fig)

    with cg2:
        st.markdown("**Top 10 Estados con Mayor Volumen de Propiedades**")
        top_vol = df_eda["state"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_vol.values, y=top_vol.index, ax=ax, color="#55a868")
        ax.set_xlabel("Cantidad de Propiedades")
        ax.set_ylabel("Estado")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Vista previa de los primeros 20 registros limpios"):
        st.dataframe(df_eda.head(20), use_container_width=True)

# ==============================================================================
# TAB 2: EXPLORACIÓN Y CORRELACIÓN
# ==============================================================================
with tab1:
    st.subheader("Distribuciones de Variables Principales")
    st.markdown(
        "El análisis visual permite validar la forma de las distribuciones, detectar sesgos y evaluar la presencia de "
        "valores extremos antes de construir los modelos predictivos."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribución del Precio (USD)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_eda["price"], bins=40, kde=True, ax=ax, color="#1f77b4")
        ax.axvline(mediana_p, color="red", linestyle="--", label=f"Mediana: ${mediana_p:,.0f}")
        ax.axvline(media_p, color="orange", linestyle=":", label=f"Media: ${media_p:,.0f}")
        ax.set_xlabel("Precio (USD)")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Distribución de la Superficie Construida (m2)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_eda["superficie_m2"], bins=40, kde=True, ax=ax, color="#2ca02c")
        ax.set_xlabel("Superficie (m2)")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Análisis de Correlación (Pearson vs. Spearman)")
    st.markdown(
        "Se calculan simultáneamente la correlación de Pearson (relaciones lineales) y la de Spearman (relaciones monótonas "
        "robustas ante asimetría y atípicos) para identificar qué características físicas presentan mayor asociación con el precio."
    )

    cols_corr = ["price", "superficie_m2", "terreno_m2", "bed", "bath", "tiene_historial_venta"]
    corr_pearson = df_eda[cols_corr].corr(method="pearson")
    corr_spearman = df_eda[cols_corr].corr(method="spearman")

    cc1, cc2 = st.columns([1.3, 1])
    with cc1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de Correlación de Pearson")
        st.pyplot(fig)
        plt.close(fig)

    with cc2:
        tabla_corr = pd.DataFrame({
            "Pearson (Lineal)": corr_pearson["price"].drop("price"),
            "Spearman (Monótona)": corr_spearman["price"].drop("price")
        }).sort_values(by="Pearson (Lineal)", ascending=False)
        st.markdown("**Correlación directa con el Precio**")
        st.dataframe(tabla_corr.style.format("{:.4f}"), use_container_width=True)

        var_top_corr = tabla_corr.index[0]
        val_top_corr = tabla_corr.iloc[0, 0]
        var_seg_corr = tabla_corr.index[1]
        val_seg_corr = tabla_corr.iloc[1, 0]
        st.info(
            f"Hallazgo de correlación: Entre las variables físicas, '{var_top_corr}' presenta la asociación lineal más fuerte "
            f"con el precio (r = {val_top_corr:.3f}), seguida por '{var_seg_corr}' (r = {val_seg_corr:.3f}). "
            f"No obstante, las correlaciones moderadas demuestran que las características físicas por sí solas no bastan "
            f"para explicar el valor total y confirman la necesidad de incluir variables de ubicación y entorno."
        )

    st.subheader("Relación Superficie Construida vs. Precio")
    muestra_plot = df_eda.sample(min(3000, len(df_eda)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=muestra_plot, x="superficie_m2", y="price", alpha=0.35, s=15, ax=ax)
    ax.set_xlabel("Superficie (m2)")
    ax.set_ylabel("Precio (USD)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "La dispersión vertical creciente a medida que aumenta la superficie evidencia heterocedasticidad: "
        "dos viviendas de igual metraje pueden diferir sustancialmente en precio según el código postal y el estado."
    )

# ==============================================================================
# TAB 3: MODELO ESTADÍSTICO (OLS)
# ==============================================================================
with tab2:
    st.subheader("Regresión Lineal Múltiple (Mínimos Cuadrados Ordinarios)")
    st.markdown(
        "El modelo econométrico de Mínimos Cuadrados Ordinarios (OLS) permite cuantificar el efecto marginal de cada variable "
        "independiente manteniendo constantes las demás, así como evaluar su significancia estadística mediante pruebas t y F."
    )

    st.latex(
        r"\hat{Y} = \beta_0 + \beta_1 X_{\text{sup}} + \beta_2 X_{\text{terr}} + \beta_3 X_{\text{bed}} + \beta_4 X_{\text{bath}} + \beta_5 X_{\text{hist}} + \beta_6 X_{\text{est}} + \beta_7 X_{\text{zip}} + \beta_8 X_{\text{oferta}} + \varepsilon"
    )

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("R2 (Entrenamiento)", f"{ols_info['r2']:.4f}")
    o2.metric("R2 Ajustado", f"{ols_info['r2_adj']:.4f}")
    o3.metric("Estadístico F", f"{ols_info['f_stat']:,.1f}")
    o4.metric("p-value Global (F)", f"{ols_info['f_pvalue']:.2e}")

    st.dataframe(
        ols_table.style.format({
            "Coeficiente": "{:,.4f}",
            "Error Estándar": "{:,.4f}",
            "t-stat": "{:,.2f}",
            "p-value": "{:.4e}"
        }),
        use_container_width=True
    )

    vars_sig = ols_table.drop("const", errors="ignore")
    n_sig = int((vars_sig["p-value"] < 0.05).sum())
    coef_sup = ols_table.loc["superficie_m2", "Coeficiente"]
    coef_zip = ols_table.loc["precio_mediano_zip", "Coeficiente"]

    st.info(
        f"Interpretación OLS: El modelo es globalmente significativo (p-value < 0.001) y explica el {ols_info['r2']*100:.1f}% "
        f"de la varianza del precio en entrenamiento. {n_sig} de las {len(vars_sig)} variables independientes resultan "
        f"estadísticamente significativas (p < 0.05). En particular, cada metro cuadrado adicional de construcción se asocia "
        f"en promedio con un cambio de ${coef_sup:,.2f} USD, mientras que el precio mediano del código postal tiene un "
        f"coeficiente de {coef_zip:.3f}, confirmando el peso determinante del entorno local."
    )

    st.subheader("Diagnóstico de Residuales (OLS)")
    idx_res = np.random.default_rng(42).choice(len(ols_info["resid"]), size=min(3000, len(ols_info["resid"])), replace=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(ols_info["fitted"][idx_res], ols_info["resid"][idx_res], alpha=0.25, s=12, color="#1f77b4")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Valores Ajustados (USD)")
    ax.set_ylabel("Residuales (USD)")
    ax.set_title("Residuales vs. Valores Ajustados")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "La apertura en forma de abanico de los residuales en propiedades de mayor valor indica varianza no constante "
        "y relaciones no lineales, lo que justifica evaluar algoritmos de Machine Learning basados en árboles."
    )

# ==============================================================================
# TAB 4: COMPARACIÓN DE MODELOS ML
# ==============================================================================
with tab3:
    st.subheader("Evaluación Comparativa en Conjunto de Prueba (20%)")
    st.markdown(
        "Todos los modelos fueron entrenados sobre el mismo conjunto de entrenamiento (80%) y evaluados sobre datos no vistos (20%) "
        "utilizando métricas de error absoluto, error cuadrático y varianza explicada."
    )

    st.latex(
        r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} \qquad \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \qquad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}"
    )

    mejor_fila = df_metricas.iloc[0]
    fila_lr = df_metricas[df_metricas["Modelo"] == "Regresión Lineal"].iloc[0]
    mejora_rmse = ((fila_lr["RMSE (USD)"] - mejor_fila["RMSE (USD)"]) / fila_lr["RMSE (USD)"]) * 100
    ganancia_r2 = (mejor_fila["R2"] - fila_lr["R2"]) * 100

    st.success(
        f"Modelo seleccionado: {mejor_fila['Modelo']} | R2 = {mejor_fila['R2']:.4f} | "
        f"RMSE = ${mejor_fila['RMSE (USD)']:,.0f} USD | MAE = ${mejor_fila['MAE (USD)']:,.0f} USD"
    )

    st.dataframe(
        df_metricas.style.format({
            "MSE": "{:,.0f}",
            "RMSE (USD)": "${:,.2f}",
            "MAE (USD)": "${:,.2f}",
            "R2": "{:.4f}"
        }),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Comparación de Score R2**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_metricas, x="Modelo", y="R2", ax=ax, color="#4c72b0")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Importancia de Variables (Random Forest)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        importancias.sort_values().plot(kind="barh", ax=ax, color="#2ca02c")
        ax.set_xlabel("Importancia Relativa")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        f"Análisis comparativo: {mejor_fila['Modelo']} supera a la Regresión Lineal reduciendo el RMSE en un "
        f"{mejora_rmse:.1f}% y aumentando la varianza explicada (R2) en {ganancia_r2:.2f} puntos porcentuales. "
        f"La superioridad de los ensambles de árboles radica en su capacidad para capturar interacciones no lineales "
        f"entre el nivel económico de la zona (código postal) y el tamaño físico de la propiedad."
    )

    modelo_vis = st.selectbox("Inspeccionar gráfica Real vs. Predicho del modelo:", df_metricas["Modelo"].tolist())
    y_pred_vis = predicciones[modelo_vis]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(y_test, y_pred_vis, alpha=0.2, s=12)
    lim_max = max(float(y_test.max()), float(y_pred_vis.max()))
    ax.plot([0, lim_max], [0, lim_max], color="red", linestyle="--", label="Predicción perfecta (y = x)")
    ax.set_xlabel("Precio Real (USD)")
    ax.set_ylabel("Precio Predicho (USD)")
    ax.set_title(f"Real vs. Predicho - {modelo_vis}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ==============================================================================
# TAB 5: SIMULADOR DE PRECIO
# ==============================================================================
with tab4:
    st.subheader("Estimador Interactivo de Valor de Propiedad")
    st.markdown(
        "Ingresa las características físicas y la ubicación de una propiedad para obtener la valoración estimada "
        "con el mejor modelo de Machine Learning frente al modelo de referencia lineal."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        estado_in = st.selectbox("Estado", sorted(df["state"].unique().tolist()))
        ciudades_estado = sorted(df[df["state"] == estado_in]["city"].unique().tolist())
        ciudad_in = st.selectbox("Ciudad", ciudades_estado)
        zips_ciudad = sorted(df[(df["state"] == estado_in) & (df["city"] == ciudad_in)]["zip_code"].unique().tolist())
        zip_in = st.selectbox("Código Postal (ZIP)", zips_ciudad)

    with col_b:
        superficie_in = st.number_input("Superficie construida (m2)", min_value=30.0, max_value=900.0, value=180.0, step=10.0)
        terreno_in = st.number_input("Superficie de terreno (m2)", min_value=50.0, max_value=80000.0, value=500.0, step=50.0)

    with col_c:
        bed_in = st.number_input("Habitaciones", min_value=1, max_value=10, value=3, step=1)
        bath_in = st.number_input("Baños", min_value=1, max_value=10, value=2, step=1)
        historial_in = st.selectbox("¿Cuenta con registro de venta previa?", ["Sí", "No"])

    p_est = refs["mediana_estado"].get(estado_in, refs["mediana_global"])
    p_zip = refs["mediana_zip"].get(zip_in, p_est)
    of_ciu = refs["oferta_ciudad"].get(ciudad_in, 1)

    entrada = pd.DataFrame([{
        "superficie_m2": superficie_in,
        "terreno_m2": terreno_in,
        "bed": bed_in,
        "bath": bath_in,
        "tiene_historial_venta": 1 if historial_in == "Sí" else 0,
        "precio_mediano_estado": p_est,
        "precio_mediano_zip": p_zip,
        "oferta_ciudad": of_ciu
    }])

    if st.button("Calcular Predicción", type="primary"):
        mejor_modelo = modelos[mejor_fila["Modelo"]]
        pred_ml = float(mejor_modelo.predict(entrada)[0])
        pred_lr = float(modelos["Regresión Lineal"].predict(entrada)[0])
        diff_zip = ((pred_ml - p_zip) / p_zip) * 100

        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"Estimación ({mejor_fila['Modelo']})", f"${pred_ml:,.0f} USD")
        r2.metric("Estimación (Regresión Lineal)", f"${pred_lr:,.0f} USD")
        r3.metric("Precio Estimado por m2", f"${pred_ml / superficie_in:,.0f} USD/m2")
        r4.metric("Referencia Mediana ZIP", f"${p_zip:,.0f} USD", f"{diff_zip:+.1f}% vs zona")

# ==============================================================================
# TAB 6: CONCLUSIONES Y RECOMENDACIONES
# ==============================================================================
with tab5:
    st.subheader("Hallazgos Principales del Estudio")

    var_imp_1 = importancias.index[0]
    val_imp_1 = importancias.iloc[0] * 100
    var_imp_2 = importancias.index[1]
    val_imp_2 = importancias.iloc[1] * 100
    var_imp_3 = importancias.index[2]
    val_imp_3 = importancias.iloc[2] * 100

    st.markdown(
        f"1. **Dominancia del entorno geográfico y las dimensiones físicas:** De acuerdo con la importancia relativa "
        f"extraída de Random Forest, las tres variables con mayor capacidad predictiva son **{var_imp_1}** ({val_imp_1:.1f}%), "
        f"**{var_imp_2}** ({val_imp_2:.1f}%) y **{var_imp_3}** ({val_imp_3:.1f}%). Esto confirma que el valor inmobiliario "
        f"está gobernado en primer lugar por el micro-mercado de ubicación y, en segundo lugar, por la superficie y capacidad física del inmueble.\n"
        f"2. **Valor explicativo vs. capacidad predictiva:** La Regresión Lineal Múltiple (OLS) demostró utilidad para inferencia "
        f"estadística y cuantificación de efectos marginales (alcanzando R2 = {fila_lr['R2']:.4f} en prueba), pero presenta "
        f"limitaciones frente a la heterocedasticidad y las relaciones no lineales del mercado.\n"
        f"3. **Selección del modelo óptimo:** **{mejor_fila['Modelo']}** obtuvo el mejor equilibrio general con "
        f"R2 = {mejor_fila['R2']:.4f}, RMSE = ${mejor_fila['RMSE (USD)']:,.0f} USD y MAE = ${mejor_fila['MAE (USD)']:,.0f} USD, "
        f"reduciendo el error cuadrático medio en un {mejora_rmse:.1f}% respecto a la regresión lineal clásica.\n"
        f"4. **Naturaleza de los datos:** El preprocesamiento y filtrado de atípicos sobre más de {meta['total_raw']:,} registros "
        f"originales fue determinante para evitar sesgos extremos por registros incompletos o propiedades de lujo atípicas."
    )

    st.subheader("Limitaciones Identificadas")
    st.markdown(
        "- El dataset público no incluye directamente variables de estado de conservación, antigüedad exacta de construcción "
        "ni índices directos de criminalidad o cercanía a servicios (escuelas, hospitales, transporte); estos factores fueron "
        "aproximados mediante indicadores agregados por código postal y estado.\n"
        "- Las predicciones en códigos postales con muy poca representación en el conjunto de entrenamiento presentan mayor incertidumbre "
        "y se respaldan en la mediana estatal como mecanismo de regularización."
    )

    st.subheader("Recomendaciones para Implementación y Mejora Continua")
    st.markdown(
        "1. **Despliegue operativo:** Implementar el modelo seleccionado como herramienta de valuación preliminar (AVM) "
        "para estimar rangos de precio sugeridos, acompañado siempre del margen de error promedio (MAE).\n"
        "2. **Enriquecimiento geoespacial:** En futuras iteraciones, cruzar el código postal (`zip_code`) con bases públicas "
        "del censo (ingreso medio por hogar, tasa de desempleo y densidad poblacional) y APIs de distancia a servicios urbanos.\n"
        "3. **Transformación logarítmica y validación cruzada:** Evaluar el entrenamiento sobre `log(price)` para reducir la influencia "
        "de la cola derecha y aplicar validación cruzada K-Fold con ajuste de hiperparámetros para producción."
    )
