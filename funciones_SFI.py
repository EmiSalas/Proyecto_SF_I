import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats
import streamlit as st
import scipy.optimize as op


@st.cache_data
def infor_ticker(tickers, fecha_inicio, fecha_fin):
    """
    Obtiene información histórica de precios para una lista de tickers utilizando Yahoo Finance.

    Parámetros:
    -----------
    tickers : list
        Lista de strings con los símbolos de los tickers
    fecha_inicio : str
        Fecha de inicio en formato 'YYYY-MM-DD'
    fecha_fin : str
        Fecha de fin en formato 'YYYY-MM-DD'

    Retorna:
    --------
    dict
        Diccionario con DataFrames de precios históricos para cada ticker
    """

    # Validar que tickers sea una lista
    if not isinstance(tickers, list):
        raise ValueError("El parámetro 'tickers' debe ser una lista")

    # Validar formato de fechas
    try:
        datetime.strptime(fecha_inicio, '%Y-%m-%d')
        datetime.strptime(fecha_fin, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Las fechas deben estar en formato 'YYYY-MM-DD'")

    datos_tickers = {}

    for ticker in tickers:
        try:
            # Descargar datos del ticker
            datos = yf.download(ticker, start=fecha_inicio, end=fecha_fin)

            if datos.empty:
                print(f"Advertencia: No se encontraron datos para {ticker}")
                continue

            # Agregar columna con el símbolo del ticker
            datos['Ticker'] = ticker

            # Guardar en el diccionario
            datos_tickers[ticker] = datos

        except Exception as e:
            print(f"Error al descargar datos para {ticker}: {str(e)}")

    return datos_tickers

def calcular_rendimientos(datos_tickers, precio_tipo='Close'):
    """
    Calcula los rendimientos diarios para cada ticker en los datos históricos.

    Parámetros:
    -----------
    datos_tickers : dict
        Diccionario con DataFrames de precios históricos (output de infor_ticker)
    precio_tipo : str
        Tipo de precio a utilizar para calcular rendimientos ('Close', 'Adj Close', 'Open', 'High', 'Low')

    Retorna:
    --------
    pandas.DataFrame
        DataFrame con los rendimientos diarios de cada ticker
    """

    # Verificar que el diccionario no esté vacío
    if not datos_tickers:
        raise ValueError("El diccionario de datos está vacío")

    # Lista para almacenar los DataFrames de rendimientos
    rendimientos_lista = []

    for ticker, datos in datos_tickers.items():
        try:
            precio_usar = precio_tipo

            # Rendimiento = (Precio_t / Precio_t-1) - 1
            rendimientos = datos[precio_usar,ticker].pct_change()

            # Crear DataFrame con los rendimientos
            rendimientos_df = pd.DataFrame({
                'Ticker': ticker,
                'Fecha': datos.index,
                'Rendimiento': rendimientos
            }).set_index('Fecha')

            rendimientos_lista.append(rendimientos_df)

        except Exception as e:
            print(f"Error al calcular rendimientos para {ticker}: {str(e)}")

    # Combinar todos los DataFrames de rendimientos
    if rendimientos_lista:
        # Concatenar todos los DataFrames
        rendimientos_completos = pd.concat(rendimientos_lista)

        # Pivotar para tener tickers como columnas
        rendimientos_pivot = rendimientos_completos.pivot_table(
            values='Rendimiento',
            index=rendimientos_completos.index,
            columns='Ticker'
        )

        # Eliminar filas con NaN (incluyendo la primera fila que tiene NaN por pct_change)
        rendimientos_limpios = rendimientos_pivot.dropna()

        return rendimientos_limpios

    else:
        raise ValueError("No se pudieron calcular rendimientos para ningún ticker")
    

    # Definición de los benchmarks preestablecidos
BENCHMARKS = {
    'Regiones': {
        'SPLG': 0.7062,
        'EWC': 0.0323,
        'IEUR': 0.1176,
        'EEM': 0.0902,
        'EWJ': 0.0537
    },
    'Sectores': {
        'XLC': 0.0999,
        'XLY': 0.1025,
        'XLP': 0.0482,
        'XLE': 0.0295,
        'XLF': 0.1307,
        'XLV': 0.0958,
        'XLI': 0.0809,
        'XLB': 0.0166,
        'XLRE': 0.0187,
        'XLK': 0.3535,
        'XLU': 0.0237
    }
}

@st.cache_data
def construir_benchmark(tipo_benchmark, fecha_inicio, fecha_fin):

    pesos = BENCHMARKS[tipo_benchmark]
    tickers = list(pesos.keys())


    # Usar infor_ticker para obtener datos históricos
    datos_historicos = infor_ticker(tickers, fecha_inicio, fecha_fin)

    # Verificar que tenemos datos para construir el benchmark
    if not datos_historicos:
        raise ValueError("No se pudieron obtener datos para ningún componente del benchmark")

    # Calcular rendimientos usando nuestra función
    rendimientos_tickers = calcular_rendimientos(datos_historicos, 'Close')

    # Calcular rendimiento del benchmark ponderado
    benchmark_rendimientos = pd.Series(0.0, index=rendimientos_tickers.index)
    componentes_utilizados = 0

    for ticker, peso in pesos.items():
        if ticker in rendimientos_tickers.columns:
            benchmark_rendimientos += rendimientos_tickers[ticker] * peso
            componentes_utilizados += 1

    # Ajustar pesos si faltan algunos componentes (normalizar a 1)
    if componentes_utilizados < len(pesos):
        peso_total_utilizado = sum(peso for ticker, peso in pesos.items()
                                 if ticker in rendimientos_tickers.columns)
        if peso_total_utilizado > 0:
            # Normalizar los pesos para que sumen 1
            benchmark_rendimientos = benchmark_rendimientos / peso_total_utilizado

    benchmark_rendimientos.name = tipo_benchmark

    return benchmark_rendimientos

@st.cache_data
def rendimientos_benchmark(tipo_benchmark, fecha_inicio, fecha_fin):

  datos_bench = infor_ticker([tipo_benchmark], fecha_inicio, fecha_fin)
  # Calcular rendimientos del benchmark
  rend_bench_df = calcular_rendimientos(datos_bench)

  # Asegurar que obtenemos la serie del benchmark
  rend_bench = rend_bench_df[tipo_benchmark]

  return rend_bench

# ============================================================
# Cálculo de Beta
# ============================================================
def calcular_beta(rendimientos_activos_df, rend_bench, tipo_benchmark):
    """
    Calcula la beta de cada activo respecto a un benchmark.
    """
    if isinstance(rendimientos_activos_df, pd.Series):
        rendimientos_activos_df = rendimientos_activos_df.to_frame(name=tipo_benchmark)

    datos = rendimientos_activos_df.join(rend_bench, how="inner", rsuffix="_bench").dropna()
    benchmark = datos[tipo_benchmark]

    betas = {}
    var_b = np.var(benchmark, ddof=1)

    for activo in rendimientos_activos_df.columns:
        cov = np.cov(datos[activo], benchmark)[0, 1]
        beta = cov / var_b if var_b != 0 else np.nan
        betas[activo] = beta

    return pd.Series(betas)


# ============================================================
# Máximo Drawdown
# ============================================================
def max_drawdown(rendimientos):
    wealth = (1 + rendimientos).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown.min()


# ============================================================
# VaR y CVaR histórico
# ============================================================
def var_historico(rendimientos, confianza=0.05):
    return np.percentile(rendimientos, confianza * 100)

def cvar_historico(rendimientos, confianza=0.05):
    var = var_historico(rendimientos, confianza)
    return rendimientos[rendimientos <= var].mean()


# ============================================================
# Sharpe Ratio
# ============================================================
def sharpe_ratio(rendimientos, rf_rate_daily=0.0, dias_anual=252):
    excess = rendimientos - rf_rate_daily
    mean_excess = excess.mean()
    std_excess = excess.std()

    if std_excess == 0:
        return np.nan

    return np.sqrt(dias_anual) * mean_excess / std_excess


# ============================================================
# Sortino Ratio (mejorado)
# ============================================================
def sortino_ratio(rendimientos, rf_rate_daily=0.0, target=0, dias_anual=252):
    excess = rendimientos - rf_rate_daily
    downside = np.minimum(0, rendimientos - target)

    # RMS downside deviation (estándar correcto)
    downside_std = np.sqrt(np.mean(downside**2))

    if downside_std == 0:
        return np.nan

    mean_excess = excess.mean()
    return np.sqrt(dias_anual) * mean_excess / downside_std


# ============================================================
# Treynor Ratio (usando beta correcta con excesos)
# ============================================================
def treynor_ratio(rendimientos, beta, tasa_libre_riesgo_anual=0.0, dias_anual=252):
    # Convertir tasa anual a diaria
    rf_daily = (1 + tasa_libre_riesgo_anual)**(1/dias_anual) - 1

    # Exceso de rendimiento diario
    excess_daily = rendimientos - rf_daily

    # Exceso de rendimiento anualizado
    excess_annual = excess_daily.mean() * dias_anual

    if beta == 0 or np.isnan(beta):
        return np.nan

    return excess_annual / beta


# ============================================================
# Information Ratio
# ============================================================
def information_ratio(rendimientos_portafolio, rendimientos_benchmark, dias_anual=252):
    active = rendimientos_portafolio - rendimientos_benchmark
    tracking_error = active.std()

    if tracking_error == 0:
        return np.nan

    return np.sqrt(dias_anual) * active.mean() / tracking_error


# ============================================================
# Calmar Ratio
# ============================================================
def calmar_ratio(rendimientos, tasa_libre_riesgo=0.0, dias_anual=252):
    cumulative = (1 + rendimientos).cumprod()
    final_value = cumulative.iloc[-1]

    cagr = final_value**(dias_anual/len(rendimientos)) - 1
    max_dd = abs(max_drawdown(rendimientos))

    if max_dd == 0:
        return np.nan

    return (cagr - tasa_libre_riesgo) / max_dd


def metricas_ticker(rendimientos_df, rendimientos_benchmark, tipo_benchmark, tasa_libre_riesgo=0.0):
    """
    Calcula todas las métricas de riesgo-rendimiento para cada ticker en el DataFrame de rendimientos.

    Parámetros:
    -----------
    rendimientos_df : pandas.DataFrame
        DataFrame con rendimientos diarios (output de calcular_rendimientos)
    rendimientos_benchmark : pandas.Series
        Rendimientos del benchmark
    tasa_libre_riesgo : float
        Tasa libre de riesgo anual (default 0.0)

    Retorna:
    --------
    pandas.DataFrame
        DataFrame con las métricas calculadas para cada ticker
    """

    # Calcular betas para todos los tickers
    betas = calcular_beta(rendimientos_df, rendimientos_benchmark,tipo_benchmark)

    dias_anual = 252

    # Convertir la tasa libre de riesgo anual a diaria
    rf_daily = (1 + tasa_libre_riesgo)**(1/252) - 1

    # Diccionario para almacenar todas las métricas
    metricas_tickers = {}

    for ticker in rendimientos_df.columns:
        rendimientos = rendimientos_df[ticker].dropna()

        # Métricas básicas
        media = rendimientos.mean() * dias_anual
        volatilidad = rendimientos.std() * np.sqrt(dias_anual)
        sesgo = rendimientos.skew()
        curtosis = rendimientos.kurtosis()

        # Métricas de riesgo
        max_dd = max_drawdown(rendimientos)
        var_95 = var_historico(rendimientos, 0.05)
        cvar_95 = cvar_historico(rendimientos, 0.05)

        # Ratios
        sharpe = sharpe_ratio(rendimientos, rf_daily)
        sortino = sortino_ratio(rendimientos, rf_daily)
        treynor = treynor_ratio(rendimientos, betas[ticker], tasa_libre_riesgo)
        calmar = calmar_ratio(rendimientos, tasa_libre_riesgo)

        # Information Ratio (vs benchmark)
        info_ratio = information_ratio(rendimientos, rendimientos_benchmark)

        metricas_tickers[ticker] = {
            'Beta': betas[ticker],
            'Media': media,
            'Volatilidad': volatilidad,
            'Sesgo': sesgo,
            'Curtosis': curtosis,
            'Max_Drawdown': max_dd,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Treynor_Ratio': treynor,
            'Information_Ratio': info_ratio,
            'Calmar_Ratio': calmar
        }

    # Convertir a DataFrame para mejor visualización
    df_metricas = pd.DataFrame(metricas_tickers).T

    return df_metricas

def preparar_parametros(rendimientos_df):
    """
    A partir del DataFrame de rendimientos calcula:
    - media anualizada
    - matriz de covarianzas anualizada
    - número de activos
    """

    mean_returns = rendimientos_df.mean() * 252       # Rendimiento anual
    cov_matrix   = rendimientos_df.cov() * 252        # Covarianza anual
    tickers      = rendimientos_df.columns.tolist()
    n            = len(tickers)

    return mean_returns, cov_matrix, tickers, n

def port_performance(weights, mean_returns, cov_matrix):
    """
    Retorno y volatilidad del portafolio.
    """
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights @ cov_matrix @ weights.T)
    return ret, vol

def port_min_var(mean_returns, cov_matrix, n):
    """
    Optimización de mínima varianza.
    """
    x0 = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def vol_min(w):
        return np.sqrt(w @ cov_matrix @ w.T)

    result = op.minimize(vol_min, x0, bounds=bounds, constraints=constraints)
    w = result.x
    ret, vol = port_performance(w, mean_returns, cov_matrix)
    return w, ret, vol

def port_max_sharpe(mean_returns, cov_matrix, n, risk_free=0.0):
    """
    Optimización del portafolio de máximo Sharpe.
    """
    x0 = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def neg_sharpe(w):
        r, vol = port_performance(w, mean_returns, cov_matrix)
        return -(r - risk_free) / vol

    result = op.minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    w = result.x
    ret, vol = port_performance(w, mean_returns, cov_matrix)
    return w, ret, vol

def port_markowitz_target(mean_returns, cov_matrix, n, target_return):
    """
    Minimiza la varianza sujeto a:
        - suma de pesos = 1
        - retorno = target_return
    """
    x0 = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
    ]

    def variance(w):
        return w @ cov_matrix @ w.T

    result = op.minimize(variance, x0, bounds=bounds, constraints=constraints)
    w = result.x
    ret, vol = port_performance(w, mean_returns, cov_matrix)
    return w, ret, vol


def normalizar_pesos(w):
    w = np.array(w, dtype=float)
    suma = w.sum()
    if suma == 0:
        return w
    return w / suma

def limpiar_pesos(w, tol=1e-6):
    return [0 if abs(x) < tol else x for x in w]


def portafolio_pesos(returns_df, weights):
    port_ret = returns_df.values @ weights
    return pd.Series(port_ret, index=returns_df.index, name='Port_Returns')
