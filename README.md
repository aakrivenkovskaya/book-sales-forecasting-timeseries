# Retail Book Sales Forecasting System  
### Weekly & Monthly Demand Forecasting • SARIMA • XGBoost • LSTM • Hybrid Models

---

## 🔒 NDA Notice

This project reproduces the forecasting pipeline originally developed for a  
**large UK book retail chain**, based on weekly point-of-sale data from an  
industry-standard retail book sales panel.

All original ISBNs, sales volumes, client names and KPIs are **protected under NDA**.  
The dataset provided here is **fully synthetic**, designed to imitate real book demand  
patterns (trend, seasonality, volatility) without revealing proprietary information.

---

## 📌 Project Overview

This system forecasts **weekly and monthly retail book sales** at the title level using  
four modelling families:

- **SARIMA (Auto ARIMA)**  
- **XGBoost (supervised time-series)**  
- **LSTM (deep learning with KerasTuner)**  
- **Hybrid SARIMA–LSTM models**

The workflow reflects realistic retail forecasting challenges:

- irregular weekly sales  
- short modern sales history (post-2012)  
- strong seasonality  
- promotion-driven spikes  
- noisy SKU-level patterns

Two representative titles were selected for full modelling and comparison.

---

## 🧠 Key Methods

### ✔ Classical Time-Series (SARIMA)
- Seasonal decomposition  
- ACF/PACF diagnostics  
- Stationarity checks  
- Auto ARIMA for model selection  
- 32-week forecast with confidence intervals  

### ✔ Machine Learning (XGBoost)
- Sliding-window supervised learning  
- Lag and rolling statistical features  
- Calendar effects  
- Cross-validation & grid search  
- 32-week forecast with MAE/MAPE evaluation  

### ✔ Deep Learning (LSTM)
- Sequence modelling  
- Hyperparameter tuning via **KerasTuner**  
- 32-week forecast for each title  

### ✔ Hybrid Models
**Sequential Hybrid**  
SARIMA → residual extraction → LSTM on residuals → combined forecast  

**Parallel Hybrid**  
Weighted ensemble of SARIMA + LSTM, including weight optimisation  

### ✔ Monthly Forecasting
- Weekly data aggregated to monthly  
- SARIMA and XGBoost compared on **8-month horizon**

---

## 🏗 Architecture Overview

 ┌──────────────────────────────────────────────┐
 │      Synthetic Retail Book Sales Data        │
 │  (weekly POS, metadata, ISBN-level details)  │
 └──────────────────────────────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │           Initial Data Investigation         │
 │  • Resampling irregular weeks                │
 │  • Filling missing periods with 0            │
 │  • Datetime index setup                      │
 │  • Title filtering and lifecycle inspection  │
 └──────────────────────────────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │            Feature Engineering               │
 │  • Lags, rolling stats, calendar variables   │
 │  • Train/validation split                    │
 └──────────────────────────────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │        Modelling Families (Weekly 32w)       │
 │  • SARIMA (Auto ARIMA)                       │
 │  • XGBoost + CV                              │
 │  • LSTM + KerasTuner                         │
 │  • Hybrid (Sequential + Parallel)            │
 └──────────────────────────────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │            Monthly Modelling (8m)            │
 │  • SARIMA                                    │
 │  • XGBoost                                   │
 └──────────────────────────────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │                  Final Outputs               │
 │  • Forecasts (weekly & monthly)              │
 │  • Confidence intervals                      │
 │  • MAE / MAPE metrics                        │
 │  • Model comparison & insights               │
 └──────────────────────────────────────────────┘

---

## 🚀 Business Impact

Accurate title-level forecasting enables:

- more stable inventory planning  
- improved replenishment cycles  
- reduced overstock and out-of-stock events  
- better promotion planning  
- informed print-run decisions  
- category-level performance forecasting  

Highly relevant for:

- book and media retail  
- publishers and distributors  
- e-commerce platforms  
- omnichannel retail operations  

---

## 🛠 Tech Stack

- Python  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- **Statsmodels** (decomposition, ACF/PACF, ADF, SARIMA)  
- **pmdarima** (Auto ARIMA)  
- **scikit-learn** (pipelines, CV, grid search)  
- **XGBoost**  
- **TensorFlow / Keras**  
- **KerasTuner**  

---

## ✨ Author

Project structure and methodology prepared for public demonstration.  
Original client work remains fully protected under NDA.
