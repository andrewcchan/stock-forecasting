import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from gsheetsdb import connect
import base_functions

# APP

keras = tf.keras

st.title('Stock Forecasting')
tic_option = st.selectbox(
    'Select Ticker Symbol',
    ('GOOG','AAPL','MSFT'))

st.write('You selected:', tic_option)


option_fore = st.selectbox(
    'Select Forcasting Algorithm',
    ('Naive', 'Difference_Moving_Average', 'RNN'))

st.write('You selected:', option_fore)

option_window_size = st.slider(
    'Select Window Averaging Size',
    365*2, 365)

st.write('You selected:', option_window_size)

st.write('## Tune Hyperparameters')

st.write('## Graph')


# LOAD DATA
# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    return rows
if tic_option == 'GOOG':
    sheet_url = st.secrets["gsheets_url_goog"]
elif tic_option == 'AAPL':
    sheet_url = st.secrets["gsheets_url_aapl"]
else:
    sheet_url = st.secrets["gsheets_url_msft"]

rows = run_query(f'SELECT * FROM "{sheet_url}"')

dates = []
prices = []
for row in rows:
  dates.append(row.Date)
  prices.append(row.Close)


split_time = 365*3-367
time_train = dates[:split_time]
x_train = prices[:split_time]
time_valid = dates[split_time:]
x_valid = prices[split_time:]

if option_fore == 'Naive':

    # PERFORM ML ALGO
    naive_forecast = prices[split_time - 1:-1]
    forecast = naive_forecast
    # PLOT
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_valid[0:365], x_valid[0:365], label='Series')
    plt.plot(time_valid[1:366], naive_forecast[1:366], label='Forecast')
    plt.legend(fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    title = tic_option + ' 1 Year Closing Price'
    # https://stackoverflow.com/a/15067854
    ax = plt.gca()
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)
elif option_fore == 'Difference_Moving_Average':
    # PERFORM ML ALGO
    time = np.array(dates)
    series = np.array(prices)
    diff_series = (series[365:] - series[:-365])
    diff_moving_avg = base_functions.moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
    diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

    forecast = diff_moving_avg_plus_past

    # PLOT
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_valid[0:365], x_valid[0:365], label='Series')
    plt.plot(time_valid[1:366], forecast[1:366], label='Forecast')
    plt.legend(fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    title = tic_option + ' 1 Year Closing Price'
    # https://stackoverflow.com/a/15067854
    ax = plt.gca()
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.title(title)
    plt.grid(True)
    st.pyplot(fig)


# DISPLAY METRICS

metric_res = keras.metrics.mean_absolute_error(x_valid, forecast).numpy()

"""
         ## Metrics
         Compare metrics between different forecasting methods
"""

st.write(f'Selected option: `{option_fore}`')
st.write('MAE is ', metric_res)

d = {'Metric Name':[],'Mean Absolute Error':[]}
df = pd.DataFrame(data = d)

df2 = pd.DataFrame([[option_fore,metric_res]],columns=['Metric Name','Mean Absolute Error'])

df = df.append(df2)

st.write(df)
