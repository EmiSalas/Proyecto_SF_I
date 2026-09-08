
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import funciones_SFI as sfi
from PIL import Image
from identidad_aq import identidad_AQ
import comen_sectores as comS
from black_litterman import modelo_black_litterman

def aumentar_tamaño_fuente():
    st.markdown("""
        <style>
        /* Aumenta el tamaño de la fuente para el texto normal (el que usa st.write) */
        p {
            font-size: 18px !important; /* Puedes probar con 16px, 18px o 1.1rem */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

aumentar_tamaño_fuente()

LOGO_PATH = "LogoAlmaQuant.png" 
img = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="Alma Quant",
    page_icon=img,
    #layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(LOGO_PATH, size='large')

st.sidebar.markdown("# Menú")

menu = st.sidebar.selectbox(
    "",
    [
        "Quiénes Somos",
        "Nuestro punto de vista", 
        "Crea tu propio portafolio", 
        "Modelo de B-L"
    ]
)

# ---------------------------
# SECCIÓN: Quiénes Somos
# ---------------------------

if "Quiénes Somos" in menu:
    identidad_AQ()

# ---------------------------
# SECCIÓN 1: Nuestro punto de vista
# ---------------------------

elif "punto de vista" in menu:

    LOGO_PATH = "AlmaQuant.png"

    # --- Logo Centrado y Título ---
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(LOGO_PATH, width=300)

        st.title('Market Insigts')

    st.markdown("---")

    sector = st.selectbox(
        "Selecciona un sector",
        list(comS.comentarios.keys())
    )

    st.header(str(sector))

    st.markdown(comS.comentarios[sector])

    perspectiva = comS.perspectivas[sector]

    st.markdown(
        f"""
        <div style="margin-bottom: 8px; font-size: 1rem;">
            <strong>{sector}</strong> — {comS.badge_perspectiva(perspectiva)}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# SECCIÓN 2: Crea tu propio portafolio
# ---------------------------

elif "portafolio" in menu:

    st.header("Crea tu propio portafolio")
    st.markdown("""
    Diseña tu portafolio seleccionando ETFs, evaluando métricas de riesgo-rendimiento  
    y comparándolo contra tres portafolios óptimos clásicos de teoría moderna.
    """)

    column1, column2, column3 = st.columns(3)

    with column1:

        # FECHAS
        st.subheader("Selección del periodo de análisis")
        st.write("""
        Elegimos el rango de fechas para descargar la información histórica de los activos.  
        Contamos con datos desde **2018 hasta hoy**.
        """)

        fecha_hoy = datetime.today()
        fecha_min = datetime(year=2018, month=1, day=2)
        fecha_max = datetime(year=fecha_hoy.year, month=fecha_hoy.month, day=fecha_hoy.day)

        fecha_inicio = st.date_input(
            "Elige la fecha inicial:",
            datetime(2018, 1, 3),
            min_value=fecha_min,
            max_value=fecha_max
        )

        fecha_inicio = fecha_inicio.strftime("%Y-%m-%d")
        fecha_max = fecha_max.strftime("%Y-%m-%d")

        st.session_state.fecha_inicio = fecha_inicio
        st.session_state.fecha_max = fecha_max

    with column2:
        # TIPO DE ESTRATEGIA
        st.subheader("Selecciona tu tipo de estrategia")
        st.markdown("""
        Puedes construir tu portafolio basado en:
        - **Sectores** de la economía (tecnología, salud, energía…)
        - **Regiones** geográficas (EE.UU., Canadá, Europa, Asia…)

        Elige una opción:
        """)

        estrategia = st.selectbox("Estrategia", ["Sectores", "Regiones"])

        etfs_sectores = ["XLB","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XLC"]
        etfs_regiones = ["SPLG","EWC","IEUR","EEM","EWJ"]

        
    with column3:


        # Selección dinámica según estrategia
        if estrategia == "Sectores":
            st.subheader("Selecciona uno o varios sectores:")
            seleccion = st.multiselect("Sectores disponibles:", etfs_sectores)
            r_bench = sfi.construir_benchmark(estrategia, fecha_inicio, fecha_max)
        else:
            st.subheader("Selecciona una o varias regiones del mundo:")
            seleccion = st.multiselect("Regiones disponibles:", etfs_regiones)
            r_bench = sfi.construir_benchmark(estrategia, fecha_inicio, fecha_max)

        st.session_state.seleccion = seleccion
        st.session_state.r_bench = r_bench

        # TASA LIBRE DE RIESGO
        st.subheader("Tasa libre de riesgo")
        rf = st.number_input(
            "Ingresa la tasa libre de riesgo (en % anual):",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            format="%.2f"
        )

        rf_decimal = rf / 100


    # ===============================================================
    # MÉTRICAS POR ACTIVO
    # ===============================================================
    st.subheader("Cálculo de métricas por activo 📊")

    st.write("""
    A continuación puedes analizar estadísticas de los activos seleccionados.  
    Las métricas se calculan utilizando la información histórica descargada 
    y comparando cada activo contra el **benchmark correspondiente a la estrategia elegida**.
    """)

    # Aviso del benchmark utilizado
    st.info(f"""
    Se utiliza como **benchmark** el índice definido en nuestro documento metodológico.  
    Para la estrategia **{estrategia}**, el benchmark aplicado es:
    """)

    pesos_benchmark = sfi.BENCHMARKS.get(estrategia, {})

    df_bench = pd.DataFrame.from_dict(
        pesos_benchmark, orient='index', columns=['Peso']
    )
    df_bench['Peso'] = df_bench['Peso'].apply(lambda x: f"{x:.2%}")

    st.dataframe(df_bench)


    datos_tickers = sfi.infor_ticker(seleccion, fecha_inicio, fecha_max) 
    rendimiento = sfi.calcular_rendimientos(datos_tickers, precio_tipo='Close') 
    metrics = sfi.metricas_ticker(rendimiento, r_bench, estrategia, tasa_libre_riesgo=rf_decimal)


    st.write("Selecciona las métricas a visualizar")
    seleccionar_todas = st.checkbox("Seleccionar todas las métricas")

    metricas_lista = [
        "Beta","Media","Volatilidad","Sesgo","Curtosis",
        "Max_Drawdown","VaR_95","CVaR_95",
        "Sharpe_Ratio","Sortino_Ratio","Treynor_Ratio",
        "Information_Ratio","Calmar_Ratio"
    ]

    col1, col2, col3 = st.columns(3)
    seleccion_usuario = {}

    with col1:
        st.markdown("**Métricas básicas**")
        seleccion_usuario["Beta"]        = st.checkbox("Beta", value=seleccionar_todas)
        seleccion_usuario["Media"]       = st.checkbox("Media", value=seleccionar_todas)
        seleccion_usuario["Volatilidad"] = st.checkbox("Volatilidad", value=seleccionar_todas)
        seleccion_usuario["Sesgo"]       = st.checkbox("Sesgo", value=seleccionar_todas)

    with col2:
        st.markdown("**Riesgo**")
        seleccion_usuario["Curtosis"]     = st.checkbox("Curtosis", value=seleccionar_todas)
        seleccion_usuario["Max_Drawdown"] = st.checkbox("Max Drawdown", value=seleccionar_todas)
        seleccion_usuario["VaR_95"]       = st.checkbox("VaR 95%", value=seleccionar_todas)
        seleccion_usuario["CVaR_95"]      = st.checkbox("CVaR 95%", value=seleccionar_todas)

    with col3:
        st.markdown("**Ratios de desempeño**")
        seleccion_usuario["Sharpe_Ratio"]      = st.checkbox("Sharpe", value=seleccionar_todas)
        seleccion_usuario["Sortino_Ratio"]     = st.checkbox("Sortino", value=seleccionar_todas)
        seleccion_usuario["Treynor_Ratio"]     = st.checkbox("Treynor", value=seleccionar_todas)
        seleccion_usuario["Information_Ratio"] = st.checkbox("Information Ratio", value=seleccionar_todas)
        seleccion_usuario["Calmar_Ratio"]      = st.checkbox("Calmar Ratio", value=seleccionar_todas)

    metricas_seleccionadas = [m for m, val in seleccion_usuario.items() if val]

    if st.button("Mostrar métricas"):
        if len(metricas_seleccionadas) == 0:
            st.warning("Selecciona al menos una métrica antes de continuar.")
        else:
            st.write("### 📊 Resultados")
            st.dataframe(metrics[metricas_seleccionadas])


    # ===============================================================
    # ASIGNACIÓN DE PESOS
    # ===============================================================
    st.subheader("Asignación de pesos del portafolio")
    st.write("""
    Asigna un peso (%) a cada ETF.  
    La suma debe ser exactamente **100%** para continuar.
    """)

    clm1,clm2 = st.columns(2)

    with clm1:

        if len(seleccion) == 0:
            st.info("Selecciona primero al menos un ETF para asignar pesos.")
        else:
            pesos = {}

            st.markdown("### Ajusta los pesos (%):")
            st.write('Puedes introduccir el peso manualmente para cifras más excatas')

            for ticker in seleccion:
                pesos[ticker] = st.number_input(
                    f"Peso para {ticker} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=5.0,
                    format="%.2f"
                )

        with clm2:

            suma_total = sum(pesos.values())

            st.markdown("### Estado del portafolio")
            if suma_total < 100:
                st.error(f"La suma de pesos es {suma_total:.2f}%. Falta por asignar.")
            elif suma_total > 100:
                st.error(f"La suma de pesos es {suma_total:.2f}%. Te pasaste.")
            else:
                st.success("¡Perfecto! Los pesos suman 100%.")

            st.progress(min(suma_total / 100, 1.0))

            if st.button("Confirmar portafolio"):
                if suma_total != 100:
                    st.warning("Debes ajustar los pesos para que sumen exactamente 100%.")
                else:
                    pesos_decimales = {t: p/100 for t, p in pesos.items()}
                    st.session_state.pesos_decimales = pesos_decimales
                    st.success("Portafolio guardado correctamente.")

    # ===============================================================
    # PORTAFOLIOS ÓPTIMOS
    # ===============================================================
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Construcción de Portafolios Óptimos")
        st.markdown("""
        Se generan y comparan tres portafolios:
        - **Mínima Varianza** (riesgo mínimo posible)
        - **Máximo Sharpe** (mejor riesgo-rendimiento)
        - **Markowitz con rendimiento objetivo** (personalizable)
        """)

    with c2:

        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        target_return = st.number_input(
            "Ingresa el rendimiento objetivo anual (%) para Markowitz:",
            min_value=0.0,
            max_value=50.0,
            step=0.1
        ) / 100


    if st.button("Generar portafolios óptimos"):

        mean_returns, cov_matrix, tickers, n = sfi.preparar_parametros(rendimiento)

        w_minvar, r_minvar, vol_minvar = sfi.port_min_var(mean_returns, cov_matrix, n)

        st.session_state.w_minvar = w_minvar

        w_sharpe, r_sharpe, vol_sharpe = sfi.port_max_sharpe(
            mean_returns, cov_matrix, n, risk_free=rf_decimal
        )

        st.session_state.w_sharpe = w_sharpe

        try:
            w_mark, r_mark, vol_mark = sfi.port_markowitz_target(
                mean_returns, cov_matrix, n, target_return
            )
            markowitz_ok = True

            st.session_state.markowitz_ok = markowitz_ok
            st.session_state.w_mark = w_mark
        except:
            markowitz_ok = False
            st.error("No fue posible generar el portafolio con ese rendimiento objetivo.")

        seleccion = st.session_state.seleccion
        pesos_decimales = st.session_state.pesos_decimales

        st.subheader("Comparación visual")

        col1, col2, col3, col4 = st.columns(4)

        def pie_portafolio_plotly(col, weights, title):
            df_plot = {"Ticker": tickers, "Peso": weights}
            fig = px.pie(
                df_plot,
                values="Peso",
                names="Ticker",
                title=title,
                hole=0.0
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            col.plotly_chart(fig, use_container_width=True)

        with col1:
            pie_portafolio_plotly(col1, pesos_decimales.values(), "Tu Portafolio")


        with col2:
            pie_portafolio_plotly(col2, w_minvar, "Mínima Varianza")

        with col3:
            w_sharpe_limpios = sfi.limpiar_pesos(w_sharpe)
            w_sharpe_norm = sfi.normalizar_pesos(w_sharpe_limpios)
            pie_portafolio_plotly(col3, w_sharpe_norm, "Máximo Sharpe")


        with col4:
            if markowitz_ok:
                w_mark_limpios = sfi.limpiar_pesos(w_mark)
                w_mark_norm = sfi.normalizar_pesos(w_mark_limpios)
                pie_portafolio_plotly(col4, w_mark_norm, "Markowitz (Target)")

            else:
                st.warning("Portafolio no disponible.")

    # ===============================================================
    # ANÁLISIS DE MÉTRICAS DE LOS PORTAFOLIOS
    # ===============================================================

    # Cargar variables persistentes
    pesos_decimales = st.session_state.pesos_decimales
    w_minvar = st.session_state.w_minvar
    w_sharpe = st.session_state.w_sharpe 
    w_mark = st.session_state.w_mark
    markowitz_ok = st.session_state.markowitz_ok

    st.subheader("Cálculo de métricas por portafolio 📊")
    st.write("""
    En esta sección puedes analizar, comparar y evaluar el comportamiento de cada portafolio 
    construido en el análisis, incluyendo el benchmark.  
    Las métricas presentadas permiten comprender tanto el rendimiento como el riesgo asociado 
    a cada estrategia, así como su relación frente al mercado de referencia.
    """)

    # Crear rendimientos para cada portafolio
    pesos_usuario = np.array(list(pesos_decimales.values()))
    ret_user = sfi.portafolio_pesos(rendimiento, pesos_usuario)
    ret_minvar = sfi.portafolio_pesos(rendimiento, w_minvar)
    ret_sharpe = sfi.portafolio_pesos(rendimiento, w_sharpe)

    # diccionario de métricas
    metricas_dict = {
        "Tu Portafolio": sfi.metricas_ticker(ret_user.to_frame("Tu Portafolio"), r_bench, estrategia, rf_decimal),
        "Min Var": sfi.metricas_ticker(ret_minvar.to_frame("Min Var"), r_bench, estrategia, rf_decimal),
        "Máximo Sharpe": sfi.metricas_ticker(ret_sharpe.to_frame("Sharpe"), r_bench, estrategia, rf_decimal)
    }

    if markowitz_ok and w_mark is not None:
        ret_mark = sfi.portafolio_pesos(rendimiento, w_mark)
        metricas_dict["Markowitz (Target)"] = sfi.metricas_ticker(ret_mark.to_frame("Markowitz"), r_bench, estrategia, rf_decimal)
        
    metricas_dict["Benchmark"] = sfi.metricas_ticker(r_bench.to_frame(name=estrategia), r_bench, estrategia, rf_decimal)


    # ===============================================================
    # VISUALIZACIÓN DE MÉTRICAS
    # ===============================================================

    st.write("Selecciona el portafolio a analizar")
    port_selected = st.selectbox("Portafolio:", list(metricas_dict.keys()))

    metricas_port_df = metricas_dict[port_selected]

    st.markdown(f"### Métricas de Rendimiento y Riesgo: **{port_selected}**")

    metricas_config = {
        "Beta": ("Beta", "num"),
	    "Media": ("Media anualizada", "pct"),
        "Volatilidad": ("Volatilidad anualizada", "pct"),
	    "Sesgo": ("Sesgo", "num"),
        "Curtosis": ("Curtosis", "num"),
	    "Max_Drawdown": ("Máximo Drawdown", "pct"),
	    "VaR_95": ("VaR 95%", "pct"),
        "CVaR_95": ("CVaR 95%", "pct"),
	    "Sharpe_Ratio": ("Sharpe Ratio", "num"),
        "Sortino_Ratio": ("Sortino Ratio", "num"),
        "Treynor_Ratio": ("Treynor Ratio", "num"),
	    "Information_Ratio": ("Information Ratio", "num"),
	    "Calmar_Ratio": ("Calmar Ratio", "num"),	
    }

    cols = st.columns(3)
    i = 0

    for key, (label, tipo) in metricas_config.items():
        if key in metricas_port_df.columns:

            valor = metricas_port_df[key].iloc[0]

            # Formato unificado
            if tipo == "pct":
                valor_str = f"{valor:.2%}"
            else:
                valor_str = f"{valor:.4f}"

            with cols[i % 3]:
                st.metric(label, valor_str)

            i += 1
    # ===============================================================
    # GRÁFICA DE SERIES DE TIEMPO DE LOS PORTAFOLIOS
    # ===============================================================

    st.subheader("Evolución histórica de los portafolios")
    st.write("""
    El siguiente gráfico muestra el crecimiento acumulado de cada portafolio a lo largo del tiempo, 
    incluyendo el benchmark utilizado para la comparación.  
    El benchmark aparece resaltado para facilitar su visualización frente a las estrategias analizadas.
    """)


    df_plot = pd.DataFrame({
        "Tu Portafolio": (1 + ret_user).cumprod(),
        "Mínima Varianza": (1 + ret_minvar).cumprod(),
        "Máximo Sharpe": (1 + ret_sharpe).cumprod(),
        "Benchmark": (1 + r_bench).cumprod()
    })

    if markowitz_ok:
        df_plot["Markowitz Target"] = (1 + ret_mark).cumprod()

    fig = px.line(
        df_plot,
        title="Crecimiento histórico de los portafolios vs Benchmark",
        labels={"index": "Fecha", "value": "Crecimiento acumulado"},
    )

    # benchmark mas visible
    fig.update_traces(
        selector=dict(name="Benchmark"),
        line=dict(width=4, dash="solid", color="black")
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# SECCIÓN 3: Modelo de B-L
# ---------------------------

elif "Modelo de B-L" in menu:
    modelo_black_litterman()
