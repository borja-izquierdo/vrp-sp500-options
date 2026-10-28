"""vrp: forecasting and trading the S&P 500 variance risk premium.

Code accompanying the Master's thesis *Arbitraje estadístico de volatilidad:
modelado del Variance Risk Premium mediante Machine Learning*
(Borja Izquierdo Piñero, Máster en Asesoramiento y Planificación Financiera,
Universidad Rey Juan Carlos, 2026).

Modules
-------
data      Data download (ThetaData, FRED, Stooq/yfinance), option-chain cleaning, local I/O.
variance  Realised-volatility estimators, model-free implied volatility, VRP target.
features  The 22 predictors (realised vol, option microstructure, macro and momentum).
models    Huber-loss gradient boosting, purged walk-forward CV, naive baselines.
backtest  Event-driven delta-hedged short straddle, performance metrics, benchmarks.
plots     Figures used in the README and the research summary.
"""

__version__ = "1.0.0"
