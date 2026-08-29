# Métricas de valuación de activos

Aplicación web en Streamlit para descargar precios de Yahoo Finance y calcular
indicadores de rentabilidad, riesgo y desempeño frente a un índice de referencia.

## Funcionalidades

- Selección dinámica de 1 a 10 activos y un benchmark.
- Precios de cierre ajustados, alineados por fecha.
- Periodicidad diaria, semanal o mensual.
- Rentabilidad y volatilidad anualizadas.
- Índices de Sharpe y Treynor.
- Correlación de Pearson, beta y significancia estadística.
- Rendimiento CAPM y alfa.
- VaR paramétrico porcentual y monetario.
- VaR de un portafolio equiponderado.
- Gráficas de precios normalizados, correlación y regresión.
- Descarga de resultados y precios en formato CSV.

## Estructura

```text
valuacion_activos_streamlit/
├── app.py
├── requirements.txt
└── README.md
```

## Instalación local

Se recomienda Python 3.11 o 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Windows, active el entorno con:

```powershell
.venv\Scripts\activate
```

## Ejecución

```bash
streamlit run app.py
```

Streamlit abrirá la aplicación en `http://localhost:8501`.

## Publicación en GitHub y Streamlit Community Cloud

1. Cree un repositorio nuevo en GitHub.
2. Cargue `app.py`, `requirements.txt` y `README.md` en la raíz.
3. Ingrese a [Streamlit Community Cloud](https://share.streamlit.io/).
4. Seleccione **Create app** y vincule el repositorio.
5. Defina `app.py` como archivo principal y publique.

La aplicación no requiere claves API. Yahoo Finance puede aplicar límites
temporales de consulta; si esto ocurre, espere unos minutos y vuelva a intentar.

## Parámetros principales

- **Tickers:** símbolos compatibles con Yahoo Finance, por ejemplo `AAPL`,
  `MSFT`, `AMXL.MX`, `^MXX` o `^GSPC`.
- **Tasa libre de riesgo:** puede capturarse manualmente para el país y horizonte
  relevantes. La opción automática usa el rendimiento de T-Bills de 13 semanas
  de Estados Unidos (`^IRX`).
- **Capital:** el VaR individual supone que todo el capital se invierte en cada
  activo. El tablero también presenta el VaR de un portafolio equiponderado.
- **Intervalo de confianza:** 90%, 95% o 99% para el cálculo del VaR.
- **Nivel de significancia:** se utiliza para evaluar el p-valor de la beta.
- **Plazo del VaR:** se expresa en días hábiles y se ajusta a la periodicidad de
  los rendimientos.

## Fórmulas

Sea `N` el número de periodos por año: 252 para datos diarios, 52 para semanales
y 12 para mensuales.

```text
Rendimiento periódico = Precio_t / Precio_(t-1) - 1
Rentabilidad anualizada = (1 + media periódica)^N - 1
Volatilidad anualizada = desviación estándar muestral × sqrt(N)
Sharpe = (rentabilidad anual - tasa libre de riesgo) / volatilidad anual
Beta = covarianza(activo, mercado) / varianza(mercado)
Treynor = (rentabilidad anual - tasa libre de riesgo) / beta
CAPM = tasa libre de riesgo + beta × (rentabilidad mercado - tasa libre de riesgo)
Alfa = rentabilidad anual - CAPM
VaR % = max(0, z × volatilidad periódica × sqrt(h) - media periódica × h)
VaR monetario = VaR % × capital
```

## Consideraciones metodológicas

- El benchmark se conserva como la última columna del bloque de precios y
  rendimientos.
- Las series se cruzan por fecha y solo se calculan métricas sobre observaciones
  comunes.
- El VaR implementado es paramétrico y supone rendimientos aproximadamente
  normales; no representa una pérdida máxima garantizada.
- Los resultados tienen fines informativos y académicos. No constituyen una
  recomendación de inversión.
