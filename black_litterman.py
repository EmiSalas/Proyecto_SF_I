import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import funciones_SFI as sfi


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Los pesos del benchmark deben sumar > 0")
    return w / s


def implied_equilibrium_returns(cov: np.ndarray, w_mkt: np.ndarray, delta: float) -> np.ndarray:
    """pi = delta * Sigma * w_mkt"""
    w = _normalize_weights(w_mkt).reshape(-1, 1)
    return (delta * cov @ w).reshape(-1, 1)


def build_PQ(tickers: list[str], views_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    views_df columns (expected):
      - Tipo: 'Absoluta' | 'Relativa'
      - Activo A
      - Activo B (solo para Relativa)
      - Q (retorno esperado, en decimales anuales)
    """
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    P_rows = []
    Q_vals = []

    for _, row in views_df.iterrows():
        tipo = str(row.get("Tipo", "")).strip()
        a = str(row.get("Activo A", "")).strip()
        b = str(row.get("Activo B", "")).strip()
        q = row.get("Q", np.nan)

        if tipo not in ("Absoluta", "Relativa"):
            continue
        if a not in idx:
            continue
        if pd.isna(q):
            continue

        p = np.zeros(n)
        if tipo == "Absoluta":
            p[idx[a]] = 1.0
        else:
            if b not in idx:
                continue
            p[idx[a]] = 1.0
            p[idx[b]] = -1.0

        P_rows.append(p)
        Q_vals.append(float(q))

    if not P_rows:
        return np.zeros((0, n)), np.zeros((0, 1))

    P = np.vstack(P_rows)
    Q = np.array(Q_vals, dtype=float).reshape(-1, 1)
    return P, Q



def validate_views(views_df: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Valida vistas para BL y regresa (df_valido, issues).

    - Filtra filas inválidas.
    - Genera mensajes humanos para mostrar en Streamlit.
    """
    issues: list[str] = []
    if views_df is None or len(views_df) == 0:
        return views_df, issues

    idx = set(tickers)
    valid_rows = []

    for i, row in views_df.iterrows():
        tipo = str(row.get("Tipo", "")).strip()
        a = str(row.get("Activo A", "")).strip()
        b = str(row.get("Activo B", "")).strip()
        q = row.get("Q", np.nan)

        row_id = f"fila {i + 1}"  # 1-indexed para usuario

        if tipo not in ("Absoluta", "Relativa"):
            issues.append(f"{row_id}: 'Tipo' inválido (usa Absoluta o Relativa).")
            continue
        if a not in idx:
            issues.append(f"{row_id}: 'Activo A' no está en el universo.")
            continue
        if pd.isna(q):
            issues.append(f"{row_id}: falta Q (%)." )
            continue

        if tipo == "Relativa":
            if not b or b not in idx:
                issues.append(f"{row_id}: en vista Relativa falta/invalid 'Activo B'.")
                continue
            if b == a:
                issues.append(f"{row_id}: en vista Relativa, Activo A y B deben ser distintos.")
                continue

        valid_rows.append(row)

    if not valid_rows:
        # Regresar df vacío con mismas columnas para no romper
        return views_df.iloc[0:0].copy(), issues

    df_valid = pd.DataFrame(valid_rows)
    df_valid = df_valid.reset_index(drop=True)
    return df_valid, issues


def omega_from_confidence(P: np.ndarray, cov: np.ndarray, tau: float, confidences: list[float], floor: float = 1e-10) -> np.ndarray:
    """
    Ω diagonal.
    Método práctico:
      var_view_k = p_k (tau Σ) p_k^T
      omega_k = var_view_k * ((1 - c) / c), con c en (0,1]
    """
    m = P.shape[0]
    if m == 0:
        return np.zeros((0, 0))

    tauSigma = tau * cov
    diag = []

    for k in range(m):
        c = confidences[k] if k < len(confidences) else 50.0
        c = max(min(float(c) / 100.0, 1.0), 1e-6)
        p = P[k:k + 1, :]
        var_view = float(p @ tauSigma @ p.T)
        omega_k = max(var_view * ((1.0 - c) / c), floor)
        diag.append(omega_k)

    return np.diag(diag)


def black_litterman_posterior(
    cov: np.ndarray,
    w_mkt: np.ndarray,
    delta: float,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regresa (pi, mu_bl, cov_bl)."""
    cov = np.asarray(cov, dtype=float)
    pi = implied_equilibrium_returns(cov, w_mkt, delta)

    if P.shape[0] == 0:
        # Sin vistas: posterior = prior
        return pi, pi, cov

    tauSigma = tau * cov
    A = P @ tauSigma @ P.T + Omega
    # Resolver sistemas lineales es más estable que invertir explícitamente
    b = (Q - P @ pi)  # (k x 1)
    x = np.linalg.solve(A, b)  # (k x 1)
    mu_bl = pi + (tauSigma @ P.T @ x)

    # Para la covarianza posterior: cov + tauSigma - tauSigma P^T A^{-1} P tauSigma
    M = np.linalg.solve(A, (P @ tauSigma))  # (k x n)
    cov_bl = cov + tauSigma - (tauSigma @ P.T @ M)
    return pi, mu_bl, cov_bl


def _benchmark_weights_for_universe(estrategia: str, tickers: list[str]) -> np.ndarray:
    """Toma pesos del BENCHMARKS y los restringe/normaliza a los tickers del universo."""
    pesos_dict = sfi.BENCHMARKS.get(estrategia, {})
    w = np.array([pesos_dict.get(t, 0.0) for t in tickers], dtype=float)
    if w.sum() <= 0:
        # fallback: igual ponderado
        w = np.ones(len(tickers), dtype=float)
    return _normalize_weights(w)


def modelo_black_litterman():
    """Sección Streamlit: Modelo Black-Litterman."""

    st.header("Modelo Black-Litterman")

    st.markdown(r"""
    Este módulo presenta la construcción de un portafolio óptimo utilizando el modelo
    **Black–Litterman**, el cual integra de manera coherente la información implícita del
    mercado con las expectativas del inversionista.

    El modelo se fundamenta en tres componentes principales:

    ---

    ### Prior del mercado

    El punto de partida del modelo corresponde a los **retornos de equilibrio implícitos**
    del mercado, los cuales se derivan del portafolio de referencia y su estructura de riesgo.

    $$
    \pi = \delta \, \Sigma \, w_{mkt}
    $$

    donde:

    - **π (pi):** vector de retornos esperados implícitos del mercado  
    - **δ (delta):** coeficiente de aversión al riesgo del mercado  
    - **Σ (Sigma):** matriz de covarianzas de los activos  
    - **wₘₖₜ:** pesos del portafolio de mercado (benchmark)

    Estos retornos no se estiman de forma subjetiva, sino que se infieren a partir de la
    composición del mercado y su nivel de riesgo.

    ---

    ### Supuesto sobre la incertidumbre de las vistas

    Se asume que la matriz de incertidumbre de las vistas, **Ω (Omega)**, es **diagonal**,
    lo que implica que las vistas del inversionista se consideran independientes entre sí.

    Este supuesto es común en aplicaciones prácticas del enfoque Black–Litterman y
    simplifica su implementación.
    """)


    # -------- Inputs base (mismo universo que parte 1) --------
    fecha_hoy = pd.Timestamp.today().to_pydatetime()
    fecha_min = pd.Timestamp(year=2018, month=1, day=2).to_pydatetime()
    fecha_max = pd.Timestamp(year=fecha_hoy.year, month=fecha_hoy.month, day=fecha_hoy.day).to_pydatetime()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Periodo")
        fecha_inicio = st.date_input("Fecha inicial", pd.Timestamp(2018, 1, 3), min_value=fecha_min, max_value=fecha_max)
        fecha_fin = st.date_input("Fecha final", fecha_max, min_value=fecha_min, max_value=fecha_max)
        fecha_inicio = pd.Timestamp(fecha_inicio).strftime("%Y-%m-%d")
        fecha_fin = pd.Timestamp(fecha_fin).strftime("%Y-%m-%d")

    with col2:
        st.subheader("Universo")
        estrategia = st.selectbox("Estrategia", ["Sectores", "Regiones"], index=0)
        etfs_sectores = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"]
        etfs_regiones = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]
        universo = etfs_sectores if estrategia == "Sectores" else etfs_regiones
        seleccion = st.multiselect("Activos a incluir (vacío = todos)", universo)
        tickers = seleccion if len(seleccion) > 0 else universo

    with col3:
        st.subheader("Parámetros B-L")
        delta = st.slider("Aversión al riesgo (δ)", min_value=0.5, max_value=10.0, value=2.5, step=0.1)
        tau = st.slider("Tau (τ)", min_value=0.001, max_value=0.25, value=0.05, step=0.001)
        rf = st.number_input(
            "Tasa libre de riesgo (% anual)",
            min_value=0.0,
            max_value=25.0,
            value=4.0,
            step=0.25,
            format="%.2f",
        )
        rf_decimal = rf / 100.0


    st.markdown("---")

    # -------- Views (opcional) --------
    st.subheader("Vistas (views) del inversionista")
    st.caption(
        "En caso de querer usar una vista active el recuadro para ingresarla. "
		"En caso de que no, deje el recuadro en blanco"
    )
	
    usar_vistas = st.checkbox("¿Deseas incorporar vistas del inversionista?", value=True)
    st.caption(
        "Si desactivas esta opción, el portafolio se construirá únicamente con el prior del mercado "
        "(sin opiniones explícitas del inversionista)."
    )

    if usar_vistas:
        with st.expander("Vistas (views) del inversionista", expanded=True):
            st.markdown("""
            Las **vistas** representan expectativas del inversionista sobre el desempeño de los activos y permiten
            personalizar el portafolio más allá de lo que sugiere el mercado.

            - **Vista absoluta:** se fija un retorno esperado para un activo específico (p. ej., “Activo A tendrá Q% anual”).
            - **Vista relativa:** se expresa que un activo tendrá un mejor desempeño que otro (p. ej., “Activo A superará a Activo B por Q% anual”).

            El parámetro de **confianza (%)** indica el grado de influencia de la vista frente a la información implícita del mercado.
            """)

            if "bl_views" not in st.session_state:
                st.session_state.bl_views = pd.DataFrame(
                    {
                        "Tipo": ["Absoluta"],
                        "Activo A": [tickers[0]],
                        "Activo B": [tickers[1] if len(tickers) > 1 else tickers[0]],
                        "Q (%)": [5.0],
                        "Confianza (%)": [60.0],
                    }
                )

            # Mantener opciones válidas si el usuario cambia el universo
            df_views = st.session_state.bl_views.copy()
            if not df_views.empty:
                df_views["Activo A"] = df_views["Activo A"].where(df_views["Activo A"].isin(tickers), tickers[0])
                df_views["Activo B"] = df_views["Activo B"].where(df_views["Activo B"].isin(tickers), tickers[0])
            st.session_state.bl_views = df_views

            edited = st.data_editor(
                st.session_state.bl_views,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Absoluta", "Relativa"], required=True),
                    "Activo A": st.column_config.SelectboxColumn("Activo A", options=tickers, required=True),
                    "Activo B": st.column_config.SelectboxColumn("Activo B", options=tickers, required=False),
                    "Q (%)": st.column_config.NumberColumn(
                        "Q (%)",
                        help="Retorno esperado de la vista (anual).",
                        min_value=-50.0,
                        max_value=50.0,
                        step=0.5,
                    ),
                    "Confianza (%)": st.column_config.NumberColumn(
                        "Confianza (%)",
                        help="Grado de confianza asignado a la vista.",
                        min_value=1.0,
                        max_value=100.0,
                        step=1.0,
                    ),
                },
            )
            st.session_state.bl_views = edited
    else:
        # Sin vistas: DataFrame vacío (el modelo utilizará únicamente el prior del mercado)
        edited = pd.DataFrame(columns=["Tipo", "Activo A", "Activo B", "Q (%)", "Confianza (%)"])

    st.markdown("---")

    # -------- Ejecutar BL --------
    if st.button("Calcular posterior Black-Litterman"):
        with st.spinner("Descargando datos y calculando Black-Litterman..."):
            datos = sfi.infor_ticker(tickers, fecha_inicio, fecha_fin)
            rend = sfi.calcular_rendimientos(datos, precio_tipo="Close")
            mean_returns, cov_matrix, tickers_calc, n = sfi.preparar_parametros(rend)

            # Asegurar orden
            cov = cov_matrix.loc[tickers_calc, tickers_calc].values

            w_mkt = _benchmark_weights_for_universe(estrategia, tickers_calc)

            # Aviso si no hay pesos benchmark para el universo seleccionado
            pesos_dict = sfi.BENCHMARKS.get(estrategia, {})
            if sum(float(pesos_dict.get(t, 0.0)) for t in tickers_calc) <= 0:
                st.info("No se encontraron pesos benchmark para este universo/subconjunto; se usarán pesos iguales como proxy de mercado.")

            # P, Q
            views_work = edited.copy()
            views_work = views_work.dropna(subset=["Tipo", "Activo A", "Q (%)", "Confianza (%)"], how="any")
            views_work["Q"] = views_work["Q (%)"].astype(float) / 100.0

            # Validar vistas (evita casos como vista relativa con A == B, B faltante, etc.)
            views_valid, issues = validate_views(views_work[["Tipo", "Activo A", "Activo B", "Q", "Confianza (%)"]], tickers_calc)
            for msg in issues:
                st.warning(msg)

            P, Q = build_PQ(tickers_calc, views_valid[["Tipo", "Activo A", "Activo B", "Q"]])

            confidences = views_valid["Confianza (%)"].astype(float).tolist()
            Omega = omega_from_confidence(P, cov, tau, confidences)

            pi, mu_bl, cov_bl = black_litterman_posterior(cov, w_mkt, delta, tau, P, Q, Omega)

            # Guardar para otras secciones
            for k in ["bl_w_minvar", "bl_w_sharpe", "bl_w_mark", "bl_mark_ok"]:
                if k in st.session_state:
                    del st.session_state[k]
                    
            st.session_state.bl_tickers = tickers_calc
            st.session_state.bl_pi = pi
            st.session_state.bl_mu = mu_bl
            st.session_state.bl_cov = cov_bl
            st.session_state.bl_rf = rf_decimal
            st.session_state.bl_estrategia = estrategia

        st.success("Listo. Posterior Black-Litterman calculado.")

        
    # -------- Mostrar resultados si existen --------
    if all(k in st.session_state for k in ["bl_tickers", "bl_mu", "bl_cov", "bl_pi"]):
        tickers_calc = st.session_state.bl_tickers
        pi = st.session_state.bl_pi
        mu_bl = st.session_state.bl_mu
        cov_bl = st.session_state.bl_cov
        rf_decimal = st.session_state.get("bl_rf", 0.0)

        st.subheader("Retornos: Prior (π) vs Posterior (μ_BL)")
        # Mostramos tanto excesos como totales
        df_mu = pd.DataFrame(
            {
                "π (exceso)": pi.reshape(-1),
                "π (Total)": pi.reshape(-1) + rf_decimal,
                "μ_BL (exceso)": mu_bl.reshape(-1),
                "μ_BL (Total)": mu_bl.reshape(-1) + rf_decimal,
            },
            index=tickers_calc,
        )
        st.dataframe(df_mu.style.format("{:.2%}"), use_container_width=True)

        st.subheader("Pesos del benchmark usados como w_mkt")
        estrategia_calc = st.session_state.get("bl_estrategia", "Sectores")
        w_mkt = _benchmark_weights_for_universe(estrategia_calc, tickers_calc)
        df_w = pd.DataFrame({"Peso": w_mkt}, index=tickers_calc)
        st.dataframe(df_w.style.format("{:.2%}"), use_container_width=True)

        st.markdown("---")
        st.subheader("Portafolios óptimos usando (μ_BL total, Σ_BL)")
        st.caption("Los portafolios se optimizan usando los retornos **Totales** de Black-Litterman (μ_BL + rf).")

        mean_bl_total = pd.Series(mu_bl.reshape(-1) + rf_decimal, index=tickers_calc)
        cov_bl_df = pd.DataFrame(cov_bl, index=tickers_calc, columns=tickers_calc)
        n = len(tickers_calc)

        c1, c2 = st.columns([1, 1])
        with c1:
            target_return = st.number_input(
                "Rendimiento objetivo anual (%) para Markowitz (BL)",
                min_value=-10.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
            ) / 100.0
        with c2:
            st.caption("Optimización long-only (0–100%) y suma de pesos = 100%.")

        if st.button("Generar portafolios BL"):
            # Usamos mean_bl_total en la optimización
            w_minvar, r_minvar, vol_minvar = sfi.port_min_var(mean_bl_total, cov_bl_df, n)
            # Pasamos rf_decimal para que el Sharpe Ratio se calcule correctamente
            w_sharpe, r_sharpe, vol_sharpe = sfi.port_max_sharpe(mean_bl_total, cov_bl_df, n, risk_free=rf_decimal)

            markowitz_ok = True
            try:
                w_mark, r_mark, vol_mark = sfi.port_markowitz_target(mean_bl_total, cov_bl_df, n, target_return)
            except Exception as e:
                markowitz_ok = False
                st.error(f"Error en Markowitz Target: {e}")
                w_mark = np.zeros(n) # Garantizamos longitud N en caso de error

            st.session_state.bl_w_minvar = np.asarray(w_minvar, dtype=float).reshape(-1)
            st.session_state.bl_w_sharpe = np.asarray(w_sharpe, dtype=float).reshape(-1)
            st.session_state.bl_w_mark = np.asarray(w_mark, dtype=float).reshape(-1)
            st.session_state.bl_mark_ok = markowitz_ok

        if all(k in st.session_state for k in ["bl_w_minvar", "bl_w_sharpe", "bl_mark_ok"]):
            w_minvar = st.session_state.bl_w_minvar
            w_sharpe = st.session_state.bl_w_sharpe
            w_mark = st.session_state.get("bl_w_mark")
            markowitz_ok = st.session_state.bl_mark_ok

            colA, colB, colC = st.columns(3)

            def _pie(col, w, title):
                if len(w) != len(tickers_calc):
                    col.warning(f"Error: Los pesos del portafolio '{title}' tienen una longitud incorrecta ({len(w)} vs {len(tickers_calc)})")
                    return
                
                df_plot = pd.DataFrame({"Ticker": tickers_calc, "Peso": np.asarray(w, dtype=float)})
                fig = px.pie(df_plot, values="Peso", names="Ticker", title=title, hole=0.0)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                col.plotly_chart(fig, use_container_width=True)

            _pie(colA, w_minvar, "BL — Mínima Varianza")
            _pie(colB, w_sharpe, "BL — Máximo Sharpe")
            if markowitz_ok and w_mark is not None and w_mark.sum() > 0:
                _pie(colC, w_mark, "BL — Markowitz (Target)")
            else:
                colC.warning("Markowitz (Target) no disponible con ese rendimiento.")
