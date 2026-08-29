"""Aplicación Streamlit para valuación y análisis de riesgo de activos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
import yfinance as yf


APP_TITLE = "Métricas de valuación de activos"
PERIODS_PER_YEAR = {"Diaria": 252, "Semanal": 52, "Mensual": 12}
INTERVALS = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
PERIOD_OPTIONS = {
    "5 días": "5d",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "Año en curso (YTD)": "ytd",
    "12 meses": "1y",
    "5 años": "5y",
}


@dataclass(frozen=True)
class AnalysisInputs:
    annual_risk_free_rate: float
    capital: float
    confidence_level: float
    significance_level: float
    var_horizon_days: int
    periodicity: str

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.periodicity]

    @property
    def var_horizon_periods(self) -> float:
        trading_days_per_period = 252 / self.periods_per_year
        return self.var_horizon_days / trading_days_per_period


def normalize_ticker(value: str) -> str:
    """Normaliza un ticker sin alterar símbolos válidos como ^, ., - o =."""
    return value.strip().upper()


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(
    tickers: tuple[str, ...], period: str, interval: str
) -> pd.DataFrame:
    """Descarga cierres ajustados de Yahoo Finance y conserva el orden solicitado."""
    raw = yf.download(
        list(tickers),
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        if "Close" not in first_level:
            raise ValueError("Yahoo Finance no devolvió precios de cierre.")
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Yahoo Finance no devolvió precios de cierre.")
        close = raw[["Close"]].copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    elif len(tickers) == 1 and list(close.columns) == ["Close"]:
        close.columns = [tickers[0]]

    available = {str(column).upper(): column for column in close.columns}
    ordered = pd.DataFrame(index=close.index)
    for ticker in tickers:
        source_column = available.get(ticker.upper())
        if source_column is not None:
            ordered[ticker] = pd.to_numeric(close[source_column], errors="coerce")

    ordered.index = pd.to_datetime(ordered.index).tz_localize(None)
    ordered.index.name = "Fecha"
    return ordered.dropna(how="all").sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def get_us_treasury_rate() -> tuple[float, str]:
    """Obtiene el último rendimiento disponible de T-Bills de 13 semanas."""
    data = yf.download(
        "^IRX", period="5d", interval="1d", auto_adjust=True, progress=False
    )
    if data.empty or "Close" not in data:
        raise ValueError("No fue posible obtener la tasa de T-Bills.")
    values = pd.to_numeric(data["Close"].squeeze(), errors="coerce").dropna()
    if values.empty:
        raise ValueError("La serie de T-Bills no contiene observaciones válidas.")
    date = pd.Timestamp(values.index[-1]).strftime("%Y-%m-%d")
    return float(values.iloc[-1]) / 100, date


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Anualiza la media aritmética: (1 + media periódica)^N - 1."""
    mean_return = float(returns.mean())
    if mean_return <= -1:
        return np.nan
    return (1 + mean_return) ** periods_per_year - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Anualiza la desviación estándar muestral."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def parametric_var(
    returns: pd.Series,
    confidence_level: float,
    horizon_periods: float,
    capital: float,
) -> tuple[float, float, float]:
    """Calcula VaR normal: max(0, z*sigma*sqrt(h) - mu*h)."""
    z_value = float(stats.norm.ppf(confidence_level))
    expected_horizon_return = float(returns.mean()) * horizon_periods
    horizon_volatility = float(returns.std(ddof=1)) * np.sqrt(horizon_periods)
    var_pct = max(0.0, z_value * horizon_volatility - expected_horizon_return)
    return z_value, var_pct, capital * var_pct


def calculate_metrics(
    returns: pd.DataFrame,
    asset_tickers: Iterable[str],
    benchmark: str,
    inputs: AnalysisInputs,
) -> pd.DataFrame:
    """Calcula indicadores de rendimiento, riesgo y desempeño por activo."""
    rows: list[dict[str, float | str | int]] = []
    benchmark_returns = returns[benchmark]
    benchmark_annual_return = annualized_return(
        benchmark_returns, inputs.periods_per_year
    )

    for ticker in asset_tickers:
        paired = returns[[ticker, benchmark]].dropna()
        if len(paired) < 3:
            continue

        asset_returns = paired[ticker]
        market_returns = paired[benchmark]
        asset_annual_return = annualized_return(
            asset_returns, inputs.periods_per_year
        )
        asset_annual_volatility = annualized_volatility(
            asset_returns, inputs.periods_per_year
        )
        market_variance = float(market_returns.var(ddof=1))
        beta = (
            float(asset_returns.cov(market_returns)) / market_variance
            if market_variance > 0
            else np.nan
        )
        correlation = float(asset_returns.corr(market_returns))
        capm = (
            inputs.annual_risk_free_rate
            + beta * (benchmark_annual_return - inputs.annual_risk_free_rate)
            if np.isfinite(beta)
            else np.nan
        )
        alpha = asset_annual_return - capm if np.isfinite(capm) else np.nan
        sharpe = (
            (asset_annual_return - inputs.annual_risk_free_rate)
            / asset_annual_volatility
            if asset_annual_volatility > 0
            else np.nan
        )
        treynor = (
            (asset_annual_return - inputs.annual_risk_free_rate) / beta
            if np.isfinite(beta) and not np.isclose(beta, 0)
            else np.nan
        )
        regression = stats.linregress(market_returns, asset_returns)
        z_value, var_pct, var_amount = parametric_var(
            asset_returns,
            inputs.confidence_level,
            inputs.var_horizon_periods,
            inputs.capital,
        )

        rows.append(
            {
                "Activo": ticker,
                "Observaciones": len(paired),
                "Rentabilidad anual": asset_annual_return,
                "Volatilidad anual": asset_annual_volatility,
                "Sharpe": sharpe,
                "Treynor": treynor,
                "Correlación": correlation,
                "Beta": beta,
                "CAPM": capm,
                "Alpha": alpha,
                "Valor z": z_value,
                "VaR %": var_pct,
                "VaR $": var_amount,
                "p-valor beta": float(regression.pvalue),
                "Beta significativa": (
                    "Sí" if regression.pvalue < inputs.significance_level else "No"
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Activo") if rows else pd.DataFrame()


def calculate_equal_weight_portfolio_var(
    asset_returns: pd.DataFrame, inputs: AnalysisInputs
) -> tuple[float, float]:
    """Calcula VaR paramétrico de un portafolio equiponderado."""
    complete = asset_returns.dropna()
    if complete.empty:
        return np.nan, np.nan
    portfolio_returns = complete.mean(axis=1)
    _, var_pct, var_amount = parametric_var(
        portfolio_returns,
        inputs.confidence_level,
        inputs.var_horizon_periods,
        inputs.capital,
    )
    return var_pct, var_amount


def format_metrics(metrics: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Aplica formatos legibles sin modificar los valores descargables."""
    percent_columns = [
        "Rentabilidad anual",
        "Volatilidad anual",
        "CAPM",
        "Alpha",
        "VaR %",
        "p-valor beta",
    ]
    formats = {column: "{:.2%}" for column in percent_columns}
    formats.update(
        {
            "Sharpe": "{:.3f}",
            "Treynor": "{:.3f}",
            "Correlación": "{:.3f}",
            "Beta": "{:.3f}",
            "Valor z": "{:.3f}",
            "VaR $": "${:,.2f}",
            "Observaciones": "{:,.0f}",
        }
    )
    return metrics.style.format(formats, na_rep="N/D")


def correlation_chart(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    figure = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#ef4444"], [0.5, "#111827"], [1, "#10b981"]],
            text=np.round(corr.values, 3),
            texttemplate="%{text}",
            colorbar_title="ρ",
        )
    )
    figure.update_layout(
        title="Matriz de correlación de rendimientos",
        paper_bgcolor="#050807",
        plot_bgcolor="#050807",
        font_color="white",
        height=520,
    )
    return figure


def regression_chart(
    returns: pd.DataFrame, asset: str, benchmark: str
) -> go.Figure:
    paired = returns[[asset, benchmark]].dropna()
    regression = stats.linregress(paired[benchmark], paired[asset])
    x_line = np.linspace(paired[benchmark].min(), paired[benchmark].max(), 100)
    y_line = regression.intercept + regression.slope * x_line

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=paired[benchmark],
            y=paired[asset],
            mode="markers",
            name="Observaciones",
            marker={"color": "#10b981", "opacity": 0.65},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Regresión (β={regression.slope:.3f})",
            line={"color": "#f8fafc", "width": 2},
        )
    )
    figure.update_layout(
        title=f"{asset} frente a {benchmark}",
        xaxis_title=f"Rendimiento de {benchmark}",
        yaxis_title=f"Rendimiento de {asset}",
        paper_bgcolor="#050807",
        plot_bgcolor="#0b1210",
        font_color="white",
        height=500,
    )
    return figure


def price_chart(prices: pd.DataFrame) -> go.Figure:
    normalized = prices.ffill().dropna(how="all")
    normalized = normalized.divide(normalized.iloc[0]).multiply(100)
    figure = go.Figure()
    palette = ["#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#f8fafc"]
    for index, column in enumerate(normalized.columns):
        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode="lines",
                name=column,
                line={"color": palette[index % len(palette)]},
            )
        )
    figure.update_layout(
        title="Evolución de precios normalizados (base 100)",
        xaxis_title="Fecha",
        yaxis_title="Índice base 100",
        paper_bgcolor="#050807",
        plot_bgcolor="#0b1210",
        font_color="white",
        height=500,
        hovermode="x unified",
    )
    return figure


def set_page_style() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: Arial, sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top right, #0b2a20 0%, #050807 42%);
            color: #ffffff;
        }
        [data-testid="stSidebar"] { background-color: #07110e; }
        h1, h2, h3 { color: #f8fafc; }
        .hero {
            border: 1px solid #14532d;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            background: linear-gradient(120deg, rgba(16,185,129,.18), rgba(5,8,7,.7));
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.2rem; }
        .hero p { color: #a7f3d0; margin: .45rem 0 0; }
        [data-testid="stMetric"] {
            border: 1px solid #14532d;
            border-radius: 12px;
            padding: .8rem;
            background-color: rgba(7,17,14,.85);
        }
        .stButton > button, .stDownloadButton > button {
            background-color: #059669;
            color: white;
            border: 0;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[list[str], str, str, str, AnalysisInputs] | None:
    with st.sidebar:
        st.header("Parámetros")
        number_of_assets = st.number_input(
            "Número de activos", min_value=1, max_value=10, value=3, step=1
        )
        default_tickers = ["AAPL", "MSFT", "NVDA"]
        asset_tickers = []
        for index in range(int(number_of_assets)):
            default = default_tickers[index] if index < len(default_tickers) else ""
            asset_tickers.append(
                normalize_ticker(
                    st.text_input(f"Ticker del activo {index + 1}", value=default)
                )
            )

        benchmark = normalize_ticker(
            st.text_input("Índice de referencia", value="^GSPC")
        )
        periodicity = st.selectbox("Periodicidad", list(PERIODS_PER_YEAR))
        period_label = st.selectbox(
            "Plazo histórico", list(PERIOD_OPTIONS), index=5
        )

        st.divider()
        st.subheader("Tasa libre de riesgo")
        risk_free_source = st.radio(
            "Fuente",
            ["Captura manual", "T-Bill EUA automática"],
            help="Para México u otro país, capture una tasa soberana comparable al horizonte del análisis.",
        )
        risk_free_rate = st.number_input(
            "Tasa anual (%)",
            min_value=-10.0,
            max_value=100.0,
            value=4.57,
            step=0.01,
            disabled=risk_free_source == "T-Bill EUA automática",
        ) / 100

        if risk_free_source == "T-Bill EUA automática":
            try:
                risk_free_rate, rate_date = get_us_treasury_rate()
                st.caption(f"^IRX al {rate_date}: {risk_free_rate:.2%}")
            except Exception as error:
                st.error(f"No se pudo consultar ^IRX: {error}")
                return None

        st.divider()
        st.subheader("VaR y significancia")
        capital = st.number_input(
            "Capital a invertir", min_value=1.0, value=1_000_000.0, step=10_000.0
        )
        confidence_level = st.select_slider(
            "Intervalo de confianza", options=[0.90, 0.95, 0.99], value=0.95
        )
        significance_level = st.selectbox(
            "Nivel de significancia", [0.01, 0.05, 0.10], index=1
        )
        var_horizon_days = st.number_input(
            "Plazo del VaR (días hábiles)", min_value=1, max_value=252, value=1
        )

        calculate = st.button("Calcular métricas", type="primary", use_container_width=True)

    if not calculate:
        return None

    if any(not ticker for ticker in asset_tickers) or not benchmark:
        st.error("Capture todos los tickers y el índice de referencia.")
        return None
    if len(set(asset_tickers)) != len(asset_tickers):
        st.error("Los tickers de los activos no deben repetirse.")
        return None
    if benchmark in asset_tickers:
        st.error("El índice de referencia debe ser distinto de los activos.")
        return None

    inputs = AnalysisInputs(
        annual_risk_free_rate=risk_free_rate,
        capital=float(capital),
        confidence_level=float(confidence_level),
        significance_level=float(significance_level),
        var_horizon_days=int(var_horizon_days),
        periodicity=periodicity,
    )
    return asset_tickers, benchmark, PERIOD_OPTIONS[period_label], INTERVALS[periodicity], inputs


def main() -> None:
    set_page_style()
    st.markdown(
        """
        <div class="hero">
          <h1>Métricas de valuación de activos</h1>
          <p>Análisis bursátil de rendimiento, riesgo sistemático y pérdida potencial.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Los resultados son informativos y académicos; no constituyen una recomendación de inversión."
    )

    selection = render_sidebar()
    if selection is None:
        st.info("Configure los parámetros y seleccione **Calcular métricas**.")
        return

    asset_tickers, benchmark, period, interval, inputs = selection
    all_tickers = tuple(asset_tickers + [benchmark])

    try:
        with st.spinner("Descargando precios y calculando indicadores..."):
            prices = download_prices(all_tickers, period, interval)
    except Exception as error:
        st.error(f"No fue posible descargar la información: {error}")
        return

    missing = [ticker for ticker in all_tickers if ticker not in prices.columns]
    if missing:
        st.error("Yahoo Finance no devolvió datos para: " + ", ".join(missing))
        return

    aligned_prices = prices.loc[:, all_tickers].dropna()
    returns = aligned_prices.pct_change(fill_method=None).dropna()
    if len(returns) < 3:
        st.error(
            "No hay observaciones suficientes. Amplíe el plazo histórico o reduzca la periodicidad."
        )
        return

    metrics = calculate_metrics(returns, asset_tickers, benchmark, inputs)
    if metrics.empty:
        st.error("No fue posible calcular métricas con las series alineadas.")
        return

    benchmark_return = annualized_return(returns[benchmark], inputs.periods_per_year)
    portfolio_var_pct, portfolio_var_amount = calculate_equal_weight_portfolio_var(
        returns[asset_tickers], inputs
    )

    first_date = aligned_prices.index.min().strftime("%d/%m/%Y")
    last_date = aligned_prices.index.max().strftime("%d/%m/%Y")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Periodo efectivo", f"{first_date}–{last_date}")
    col2.metric("Observaciones", f"{len(returns):,}")
    col3.metric(f"Rendimiento {benchmark}", f"{benchmark_return:.2%}")
    col4.metric("VaR portafolio equiponderado", f"${portfolio_var_amount:,.2f}")
    st.caption(
        f"VaR del portafolio: {portfolio_var_pct:.2%} del capital, a {inputs.confidence_level:.0%} "
        f"de confianza y {inputs.var_horizon_days} día(s) hábil(es)."
    )

    tabs = st.tabs(["Indicadores", "Precios", "Correlación", "Regresión", "Datos"])
    with tabs[0]:
        st.subheader("Resultados por activo")
        st.dataframe(format_metrics(metrics), use_container_width=True)
        st.caption(
            "El VaR por activo supone que la totalidad del capital se invierte en ese activo. "
            "La beta se considera significativa cuando su p-valor es menor al nivel seleccionado."
        )
        st.download_button(
            "Descargar indicadores CSV",
            metrics.to_csv(index=True).encode("utf-8-sig"),
            file_name="metricas_valuacion.csv",
            mime="text/csv",
        )

    with tabs[1]:
        st.plotly_chart(price_chart(aligned_prices), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(correlation_chart(returns), use_container_width=True)

    with tabs[3]:
        selected_asset = st.selectbox("Activo", asset_tickers)
        st.plotly_chart(
            regression_chart(returns, selected_asset, benchmark),
            use_container_width=True,
        )

    with tabs[4]:
        display_prices = aligned_prices.reset_index()
        st.dataframe(display_prices, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar precios alineados CSV",
            display_prices.to_csv(index=False).encode("utf-8-sig"),
            file_name="precios_alineados.csv",
            mime="text/csv",
        )

    with st.expander("Metodología y fórmulas"):
        st.markdown(
            r"""
            - Rendimiento periódico: $r_t=P_t/P_{t-1}-1$.
            - Rentabilidad anualizada: $(1+\bar r)^N-1$.
            - Volatilidad anualizada: $s_r\sqrt{N}$.
            - Sharpe: $(R_i-R_f)/\sigma_i$.
            - Beta: $Cov(r_i,r_m)/Var(r_m)$.
            - Treynor: $(R_i-R_f)/\beta_i$.
            - CAPM: $R_f+\beta_i(R_m-R_f)$.
            - Alfa: $R_i-R_{CAPM}$.
            - VaR paramétrico: $max(0,z\sigma\sqrt{h}-\mu h)\times Capital$.

            $N$ es 252, 52 o 12 según la periodicidad. La volatilidad utiliza la
            desviación estándar muestral. Las series se alinean por fecha y el
            benchmark permanece como la última columna del bloque de datos.
            """
        )


if __name__ == "__main__":
    main()
