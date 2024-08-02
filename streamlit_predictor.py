import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import plotly.express as px

import requests_cache

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.evaluation import Evaluator

FUNDAMENTALS = {
    "sector": "Sector", 
    "shortRatio": "Short Ratio", 
    "profitMargins": "Profit Margins",
    "totalDebt": "Total Debt",
    "totalRevenue": "Total Revenue",
    "returnOnAssets": "Return On Assets",
    "forwardPE": "Price/Earnings Ratio"
    }


STYLES = {
    "Line": st.line_chart,
    "Area": st.area_chart,
    "Scatter": st.scatter_chart
}

class StockPredictor:
    def __init__(self):
        s = pd.read_csv("SP500.csv")
        session = requests_cache.CachedSession('yfinance.cache')
        session.headers['User-agent'] = 'my-program/1.0'
        self.symbols = s.Symbol.values
        self.volatility = yf.Ticker('^VIX', session=session)
        self.session=session

    def get_ticker_info(self, tickers: str) -> Tuple[Dict, pd.DataFrame] :
        t = yf.Ticker(tickers, session=self.session)
        hist = t.history(period='max', interval="1wk").drop(['Stock Splits', 'Dividends'], axis=1)
        hist.index = pd.to_datetime(hist.index).strftime('%Y-%m-%d')
        vol = self.volatility.history(period='max', interval="1wk")
        vol.index = pd.to_datetime(vol.index).strftime('%Y-%m-%d')
        vol.rename(columns={"Open":"Volatility"}, inplace=True)
        hist = hist.join(vol.Volatility).dropna()
        returns = hist.Open - hist.Close
        hist.drop(hist.index[0], inplace=True)
        returns.drop(returns.index[-1], inplace=True)
        hist['Returns'] = returns.values
        
        return t.info, hist
    
    def train_test_split(self, data:pd.DataFrame, split:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        num_train = int(len(data) * (split/100))
        return data[:num_train], data[num_train: -1]

    



def main():
    st.title('Stock Predictor App')
    st.write('This site leverages autoregressive and machine learning models to forecast the price of your favorite stocks')
    sp = StockPredictor()
    with st.sidebar:
        tickers = st.selectbox("Choose Ticker Symbol", sp.symbols)
        if not tickers:
            st.error("Please select at least one country.")
    
    info, data = sp.get_ticker_info(tickers)
    fundamentals = {val: info[key] for key, val in FUNDAMENTALS.items()}
    st.write(f'### Information for {info["longName"]} ({info["symbol"]})')
    with st.expander(label="Business Summary", expanded=False):
        st.write(info["longBusinessSummary"])

    with st.expander(label="Fundamentals", expanded=False):
        st.table(fundamentals)
    
    with st.sidebar:
        features = st.multiselect("Select Variables to plot", data.columns, default="Close")
        benchmark = st.selectbox("Compare to Benchmark", np.concatenate([sp.symbols, np.array(["SPY"])]), index=None)
        normalize = st.toggle("Normalize", value=True)
        style = st.radio("Select Style", ["Line", "Area", "Scatter"])

    graph = STYLES[style]

    if benchmark:
        _, b_data = sp.get_ticker_info(benchmark)
        data["Benchmark"] = b_data.Close
        features.append("Benchmark")
    
    if len(features) > 1 and normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = data[features]
        data_scaled[features] = min_max_scaler.fit_transform(data[features])
        graph(data_scaled, y=features)
    else:
        graph(data, y=features)



    st.write('### Univariate Autoregression')
    col1, col2, col3 = st.columns(3)
    with col1:
        op = st.selectbox("Model", ["ARMA", "ARIMA", "SARIMAX"])
        model = SARIMAX if op == "SARIMAX" else ARIMA
        diff = 0 if op == "ARMA" else 2
    
    with col2:
        predictor = st.selectbox("Predictor Variable", data.columns)

    with col3:
        p = st.number_input("Lag", 1, 100)

    s = st.slider("Train/Test Percentage", 0, 100, 80)
    train, test = sp.train_test_split(data, s)
    train["Type"] = "Train"
    test["Type"] = "Test"

    if st.button("Predict!", type="primary"):
        with st.spinner('Predicting...'):
            ARIMAmodel = model(train[predictor], order = (p, diff, 2))
            ARIMAmodel = ARIMAmodel.fit()
            y_pred = ARIMAmodel.get_forecast(len(test[predictor].index))
            y_pred_df = y_pred.conf_int(alpha = 0.05)
            y_pred_df[predictor] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
            y_pred_df.index = test.index
            y_pred_df["Type"] = "Prediction"
        graph(pd.concat([train, test, y_pred_df]), y=predictor, color="Type")
        st.write(f"Root Mean Squared Error: {root_mean_squared_error(test[predictor].values, y_pred_df[predictor].values)}")
        st.write(f"R2 Score: {r2_score(test[predictor].values, y_pred_df[predictor].values)}")

    else:
       graph(pd.concat([train, test]), y=predictor, color="Type")



    st.write('### Multivariate Autoregression')
    col1, col2, col3 = st.columns(3)
    with col1:
        predictors = st.multiselect("Predictor Variables", data.columns, default=['Close', 'Volume'])
        if not predictors:
            st.error("Please select at least two variables.")

    with col2:
        target = st.selectbox("Target Variables", data.columns)
       

    with col3:
        p = st.number_input("Lag", 1, 100, key=2)

    report = st.checkbox("Include Report")
    s = st.slider("Train/Test Percentage", 0, 100, 80, key=3)
    train, test = sp.train_test_split(data, s)
    train["Type"] = "Train"
    test["Type"] = "Test"

    predictors.append(target)
    
    if st.button("Predict!", type="primary", key=4):
        with st.spinner('Predicting...'):
            VARModel = VAR(train[predictors].diff().dropna().values)
            results = VARModel.fit(maxlags=p, ic='aic')
            pred = results.forecast(train[predictors].values, len(test.index))
            y_pred_df=pd.DataFrame(data=pred, index=test.index, columns=predictors)
            y_pred_df = pd.concat([train[predictors][-1:], y_pred_df]).sort_index().cumsum().drop(pd.Timestamp(y_pred_df.index[0]).strftime('%Y-%m-%d'))
            y_pred_df["Type"] = "Prediction"
        graph(pd.concat([train, test, y_pred_df]), y=target, color="Type")
        if report: st.write(results.summary())
        st.write(f"Root Mean Squared Error: {root_mean_squared_error(test[target].values, y_pred_df[target].values)}")
        st.write(f"R2 Score: {r2_score(test[target].values, y_pred_df[target].values)}")
    else:
        graph(pd.concat([train, test]), y=target, color="Type")



    st.write("### Deep Learning")
    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox("Target Variable", data.columns, key=7)
    with col2:
        epochs = st.number_input("Max Epochs", 1, 10, key=9)

    data.index = pd.DatetimeIndex(data.index)
    dataset = PandasDataset(data, target=target)
    s = st.slider("Train/Test Percentage", 0, 100, 80, key=10)
    train, test = sp.train_test_split(data, s)
    train["Type"] = "Train"
    test["Type"] = "Test"

    training_data, test_gen = split(dataset, offset=-len(test.index))
    test_data = test_gen.generate_instances(prediction_length=len(test.index), windows=1)

    if st.button("Predict!", type="primary", key=6):
        with st.spinner('Predicting...'):
            # Train the model and make predictions
            model = DeepAREstimator(
                prediction_length=len(test.index), freq="W", trainer_kwargs={"max_epochs": epochs}
            ).train(training_data)
            fig = plt.figure()
            forecasts = list(model.predict(test_data.input))
            plt.plot(data[[target]], color="black")
            for forecast in forecasts:
                forecast.plot()
            st.pyplot(fig)
            # st.write(test_data)
            # graph(np.concatenate(data[[target]], forecasts))
            # evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
            # agg_metrics, item_metrics = evaluator(test_data.dataset, forecasts)
            # st.write((json.dumps(agg_metrics, indent=4)))
    else:
        graph(pd.concat([train, test]), y=target, color="Type")

if __name__ == "__main__":
    main()